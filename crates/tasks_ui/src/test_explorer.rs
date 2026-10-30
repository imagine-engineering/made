use std::{
    collections::{HashMap, HashSet},
    sync::Arc,
};

#[cfg(any(feature = "rust-test-actions", feature = "jvm-test-actions"))]
use anyhow::Context as _;
use db::kvp::KeyValueStore;
use feature_flags::{FeatureFlagAppExt as _, FeatureFlagStore};
#[cfg(any(feature = "rust-test-actions", feature = "jvm-test-actions"))]
use futures::StreamExt as _;
use gpui::{
    Action, Anchor, App, AsyncWindowContext, Context, DismissEvent, Entity, EventEmitter,
    FocusHandle, Focusable, IntoElement, Pixels, Point, Render, Role, ScrollStrategy, Subscription,
    TaskExt as _, UpdateGlobal as _, WeakEntity, Window, actions, anchored, deferred,
};
use language_tools::language_tool_tree::{
    self, LanguageToolNode, LanguageToolNodeId, LanguageToolProviderStatus, LanguageToolSnapshot,
    LanguageToolTreeHost, LanguageToolTreeStatus, language_tool_tree, status_message,
};
#[cfg(any(feature = "rust-test-actions", feature = "jvm-test-actions"))]
use project::TaskSourceKind;
#[cfg(feature = "jvm-test-actions")]
use project::jvm_test_provider::JVM_JUNIT_PROVIDER_ID;
#[cfg(feature = "jvm-test-actions")]
use project::lsp_store::LspStoreEvent;
#[cfg(feature = "rust-test-actions")]
use project::rust_test_provider::{
    MAX_RUST_TEST_RERUNS, RUST_TEST_PROVIDER_ID, RustTestAction, RustTestActionPlan,
};
#[cfg(feature = "rust-test-actions")]
use project::task_store::PendingExperimentalFeatureOperation;
use project::{
    Project, ProjectPath,
    structured_execution::{
        DiscoveryGeneration, StructuredDebugCapability, StructuredExecutionEvent,
        StructuredExecutionStoreEvent, StructuredExecutionSummary, StructuredNode,
        StructuredNodeId, StructuredNodeKind, StructuredNodeState, StructuredProviderId,
        StructuredProviderSnapshot, StructuredProviderStatus, StructuredRun, StructuredRunId,
        StructuredRunPhase,
    },
};
use serde::{Deserialize, Serialize};
use settings::{DockSide, Settings, SettingsStore, TestsPanelSettingsContent};
use ui::{ContextMenu, IconName, Tooltip, prelude::*};
use ui_input::{ErasedEditorEvent, InputField};
use util::ResultExt as _;
#[cfg(any(feature = "rust-test-actions", feature = "jvm-test-actions"))]
use workspace::notifications::NotifyResultExt as _;
use workspace::{
    Panel, Workspace,
    dock::{DockPosition, PanelEvent},
    workspace_scoped_state_key,
};

actions!(
    test_explorer,
    [
        ToggleTestsPanel,
        RunSelectedTests,
        DebugSelectedTests,
        CancelTestRun,
        RerunFailedTests,
        RevealTestTerminal,
    ]
);

const TESTS_PANEL_KEY: &str = "TestsPanel";
const TESTS_PANEL_STATE_KEY: &str = "tests-panel-state-v1";
const TESTS_PANEL_STATE_VERSION: u32 = 1;
const MAX_FILTER_BYTES: usize = 256;
const MAX_TREE_DEPTH: usize = 128;

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
pub enum TestStatusFilter {
    #[default]
    All,
    Failed,
    Passed,
    Skipped,
    Running,
    Cancelled,
}

impl TestStatusFilter {
    fn matches(self, state: Option<StructuredNodeState>) -> bool {
        match self {
            Self::All => true,
            Self::Failed => state == Some(StructuredNodeState::Failed),
            Self::Passed => state == Some(StructuredNodeState::Passed),
            Self::Skipped => state == Some(StructuredNodeState::Skipped),
            Self::Running => matches!(
                state,
                Some(StructuredNodeState::Queued | StructuredNodeState::Running)
            ),
            Self::Cancelled => state == Some(StructuredNodeState::Cancelled),
        }
    }

    fn label(self) -> &'static str {
        match self {
            Self::All => "All",
            Self::Failed => "Failed",
            Self::Passed => "Passed",
            Self::Skipped => "Skipped",
            Self::Running => "Running",
            Self::Cancelled => "Cancelled",
        }
    }
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct TestExplorerFilter {
    pub query: String,
    pub status: TestStatusFilter,
}

impl TestExplorerFilter {
    fn normalized_query(&self) -> String {
        self.query.trim().to_lowercase()
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TestExplorerSelection {
    pub provider_id: StructuredProviderId,
    pub discovery_generation: DiscoveryGeneration,
    pub node_id: StructuredNodeId,
    pub run_id: Option<StructuredRunId>,
}

pub trait TestExplorerActionDelegate: Send + Sync {
    fn discover(
        &self,
        _project: &Entity<Project>,
        _cx: &mut App,
    ) -> Vec<(
        StructuredProviderId,
        gpui::Task<anyhow::Result<StructuredProviderSnapshot>>,
    )> {
        Vec::new()
    }

    fn run(
        &self,
        selection: &TestExplorerSelection,
        workspace: &WeakEntity<Workspace>,
        window: &mut Window,
        cx: &mut App,
    ) -> anyhow::Result<()>;

    fn cancel(
        &self,
        selection: &TestExplorerSelection,
        workspace: &WeakEntity<Workspace>,
        window: &mut Window,
        cx: &mut App,
    ) -> anyhow::Result<()>;

    fn debug(
        &self,
        selection: &TestExplorerSelection,
        workspace: &WeakEntity<Workspace>,
        window: &mut Window,
        cx: &mut App,
    ) -> anyhow::Result<()>;

    fn rerun_failed(
        &self,
        selections: &[TestExplorerSelection],
        workspace: &WeakEntity<Workspace>,
        window: &mut Window,
        cx: &mut App,
    ) -> anyhow::Result<()>;

    fn reveal_terminal(
        &self,
        selection: &TestExplorerSelection,
        workspace: &WeakEntity<Workspace>,
        window: &mut Window,
        cx: &mut App,
    ) -> anyhow::Result<()>;
}

#[cfg(not(any(feature = "rust-test-actions", feature = "jvm-test-actions")))]
struct NoopTestExplorerDelegate;

#[cfg(not(any(feature = "rust-test-actions", feature = "jvm-test-actions")))]
impl TestExplorerActionDelegate for NoopTestExplorerDelegate {
    fn run(
        &self,
        _: &TestExplorerSelection,
        _: &WeakEntity<Workspace>,
        _: &mut Window,
        _: &mut App,
    ) -> anyhow::Result<()> {
        anyhow::bail!("No structured test provider can run this selection")
    }

    fn cancel(
        &self,
        _: &TestExplorerSelection,
        _: &WeakEntity<Workspace>,
        _: &mut Window,
        _: &mut App,
    ) -> anyhow::Result<()> {
        anyhow::bail!("No structured test run is active for this selection")
    }

    fn debug(
        &self,
        _: &TestExplorerSelection,
        _: &WeakEntity<Workspace>,
        _: &mut Window,
        _: &mut App,
    ) -> anyhow::Result<()> {
        anyhow::bail!("No structured test provider can debug this selection")
    }

    fn rerun_failed(
        &self,
        _: &[TestExplorerSelection],
        _: &WeakEntity<Workspace>,
        _: &mut Window,
        _: &mut App,
    ) -> anyhow::Result<()> {
        anyhow::bail!("No failed structured tests are available to rerun")
    }

    fn reveal_terminal(
        &self,
        _: &TestExplorerSelection,
        _: &WeakEntity<Workspace>,
        _: &mut Window,
        _: &mut App,
    ) -> anyhow::Result<()> {
        anyhow::bail!("No task terminal is available for this selection")
    }
}

fn default_test_explorer_delegate() -> Arc<dyn TestExplorerActionDelegate> {
    #[cfg(all(feature = "rust-test-actions", feature = "jvm-test-actions"))]
    {
        Arc::new(CombinedTestExplorerDelegate)
    }
    #[cfg(all(feature = "rust-test-actions", not(feature = "jvm-test-actions")))]
    {
        Arc::new(RustTestExplorerDelegate)
    }
    #[cfg(all(feature = "jvm-test-actions", not(feature = "rust-test-actions")))]
    {
        Arc::new(JvmTestExplorerDelegate)
    }
    #[cfg(not(any(feature = "rust-test-actions", feature = "jvm-test-actions")))]
    {
        Arc::new(NoopTestExplorerDelegate)
    }
}

#[cfg(feature = "jvm-test-actions")]
struct JvmTestExplorerDelegate;

#[cfg(feature = "jvm-test-actions")]
impl TestExplorerActionDelegate for JvmTestExplorerDelegate {
    fn discover(
        &self,
        project: &Entity<Project>,
        cx: &mut App,
    ) -> Vec<(
        StructuredProviderId,
        gpui::Task<anyhow::Result<StructuredProviderSnapshot>>,
    )> {
        let store = project.read(cx).jvm_test_provider_store().cloned();
        let discovery = match store {
            Some(store) => store.update(cx, |store, cx| store.refresh(cx)),
            None if project.read(cx).remote_client().is_some() => {
                let message = if project.read(cx).is_disconnected(cx) {
                    "Reconnect to the project host to load Java tests"
                } else {
                    "The remote project host does not yet support Java test discovery and actions"
                };
                gpui::Task::ready(Err(anyhow::anyhow!(message)))
            }
            None => return Vec::new(),
        };
        vec![(
            StructuredProviderId(JVM_JUNIT_PROVIDER_ID.to_owned()),
            discovery,
        )]
    }

    fn run(
        &self,
        selection: &TestExplorerSelection,
        workspace: &WeakEntity<Workspace>,
        window: &mut Window,
        cx: &mut App,
    ) -> anyhow::Result<()> {
        dispatch_java_test_action(selection, None, workspace, window, cx)
    }

    fn cancel(
        &self,
        selection: &TestExplorerSelection,
        workspace: &WeakEntity<Workspace>,
        _: &mut Window,
        cx: &mut App,
    ) -> anyhow::Result<()> {
        require_jvm_provider(selection)?;
        let run_id = selection
            .run_id
            .as_ref()
            .context("The selected Java test has no active run")?;
        let workspace = workspace
            .upgrade()
            .context("The workspace is no longer available")?;
        let project = workspace.read(cx).project();
        let provider = project
            .read(cx)
            .jvm_test_provider_store()
            .cloned()
            .context("The project host does not support Java test actions")?;
        provider.update(cx, |provider, cx| provider.cancel_run(run_id, cx))
    }

    fn debug(
        &self,
        selection: &TestExplorerSelection,
        _: &WeakEntity<Workspace>,
        _: &mut Window,
        _: &mut App,
    ) -> anyhow::Result<()> {
        require_jvm_provider(selection)?;
        anyhow::bail!("Selected Java test Debug is not connected to the Java adapter yet")
    }

    fn rerun_failed(
        &self,
        selections: &[TestExplorerSelection],
        workspace: &WeakEntity<Workspace>,
        window: &mut Window,
        cx: &mut App,
    ) -> anyhow::Result<()> {
        anyhow::ensure!(
            selections
                .iter()
                .all(|selection| selection.provider_id.0 == JVM_JUNIT_PROVIDER_ID),
            "The failed selection does not belong to the Java test provider"
        );
        let [selection] = selections else {
            anyhow::bail!(
                "The Java test provider can rerun one current failed case at a time; select a case to run it directly"
            );
        };
        let failed_run_id = selection
            .run_id
            .as_ref()
            .context("The failed Java test has no owning run")?;
        dispatch_java_test_action(selection, Some(failed_run_id), workspace, window, cx)
    }

    fn reveal_terminal(
        &self,
        selection: &TestExplorerSelection,
        workspace: &WeakEntity<Workspace>,
        window: &mut Window,
        cx: &mut App,
    ) -> anyhow::Result<()> {
        require_jvm_provider(selection)?;
        let run_id = selection
            .run_id
            .as_ref()
            .context("The selected Java test has no retained run")?;
        let workspace = workspace
            .upgrade()
            .context("The workspace is no longer available")?;
        let project = workspace.read(cx).project();
        let provider = project
            .read(cx)
            .jvm_test_provider_store()
            .cloned()
            .context("The project host does not support Java test actions")?;
        provider.update(cx, |provider, cx| {
            provider.reveal_run_terminal(run_id, window, cx)
        })
    }
}

#[cfg(feature = "jvm-test-actions")]
fn require_jvm_provider(selection: &TestExplorerSelection) -> anyhow::Result<()> {
    anyhow::ensure!(
        selection.provider_id.0 == JVM_JUNIT_PROVIDER_ID,
        "The selected structured provider is not a Java test provider"
    );
    Ok(())
}

#[cfg(feature = "jvm-test-actions")]
fn dispatch_java_test_action(
    selection: &TestExplorerSelection,
    failed_run_id: Option<&StructuredRunId>,
    workspace: &WeakEntity<Workspace>,
    window: &mut Window,
    cx: &mut App,
) -> anyhow::Result<()> {
    require_jvm_provider(selection)?;
    let workspace_entity = workspace
        .upgrade()
        .context("The workspace is no longer available")?;
    let project = workspace_entity.read(cx).project().clone();
    let provider = project
        .read(cx)
        .jvm_test_provider_store()
        .cloned()
        .context("The project host does not support Java test actions")?;
    if provider.read(cx).is_remote() {
        return dispatch_remote_java_test_action(
            selection,
            failed_run_id,
            workspace_entity,
            provider,
            window,
            cx,
        );
    }
    let failed_run_id = failed_run_id.cloned();
    let plan = provider.read_with(cx, |provider, cx| match &failed_run_id {
        Some(run_id) => provider.plan_failed_case(
            &selection.node_id,
            selection.discovery_generation,
            run_id,
            cx,
        ),
        None => provider.plan_selected_case(&selection.node_id, selection.discovery_generation, cx),
    })?;
    let report_version = provider.update(cx, |provider, cx| {
        provider.capture_report_version(&plan, cx)
    });
    let task_contexts = workspace_entity.update(cx, |workspace, cx| {
        crate::task_contexts(workspace, window, cx)
    });
    let workspace = workspace_entity.downgrade();
    window
        .spawn(cx, async move |mut cx| {
            let Some(report_version) = report_version
                .await
                .notify_workspace_async_err(workspace.clone(), &mut cx)
            else {
                return anyhow::Ok(());
            };
            let task_contexts = task_contexts.await;
            let result = (|| {
                let task_context = task_contexts
                    .task_context_for_worktree_id(plan.worktree_id)
                    .context("No task context exists for the selected Java test worktree")?;
                let source = TaskSourceKind::groovy_jvm_test(plan.worktree_id);
                let mut resolved = plan
                    .template
                    .resolve_task(&source.to_id_base(), task_context)
                    .context("The selected Java test Task could not be resolved")?;
                resolved.resolved.show_rerun = false;
                let handle = workspace.update_in(cx, |workspace, window, cx| {
                    let current_plan =
                        provider.read_with(cx, |provider, cx| match &failed_run_id {
                            Some(run_id) => provider.plan_failed_case(
                                &plan.node_id,
                                plan.discovery_generation,
                                run_id,
                                cx,
                            ),
                            None => provider.plan_selected_case(
                                &plan.node_id,
                                plan.discovery_generation,
                                cx,
                            ),
                        })?;
                    anyhow::ensure!(
                        current_plan == plan,
                        "The selected Java test changed before its Task was scheduled"
                    );
                    anyhow::Ok(workspace.schedule_resolved_task_with_structured_handle(
                        source, resolved, true, window, cx,
                    ))
                })??;
                let registered = provider.update(cx, |provider, cx| {
                    provider.register_run_handle(plan, report_version, handle.clone(), cx)
                });
                if let Err(error) = registered {
                    if let Err(cancel_error) = cx.update(|_, cx| handle.cancel(cx)) {
                        log::warn!(
                            "Failed to cancel an untracked Java test Task: {cancel_error:#}"
                        );
                    }
                    return Err(error);
                }
                anyhow::Ok(())
            })();
            if let Err(error) = result {
                Err::<(), _>(error).notify_workspace_async_err(workspace, &mut cx);
            }
            anyhow::Ok(())
        })
        .detach();
    Ok(())
}

#[cfg(feature = "jvm-test-actions")]
fn dispatch_remote_java_test_action(
    selection: &TestExplorerSelection,
    failed_run_id: Option<&StructuredRunId>,
    workspace_entity: Entity<Workspace>,
    provider: Entity<project::jvm_test_provider::JvmTestProviderStore>,
    window: &mut Window,
    cx: &mut App,
) -> anyhow::Result<()> {
    let plan = provider.update(cx, |provider, cx| {
        provider.plan_remote_action(
            &selection.node_id,
            selection.discovery_generation,
            failed_run_id,
            cx,
        )
    });
    let task_contexts = workspace_entity.update(cx, |workspace, cx| {
        crate::task_contexts(workspace, window, cx)
    });
    let workspace = workspace_entity.downgrade();
    window
        .spawn(cx, async move |mut cx| {
            let Some(plan) = plan
                .await
                .notify_workspace_async_err(workspace.clone(), &mut cx)
            else {
                return anyhow::Ok(());
            };
            let task_contexts = task_contexts.await;
            let mut registered = false;
            let result = (|| {
                let task_context = task_contexts
                    .task_context_for_worktree_id(plan.worktree_id)
                    .context("No task context exists for the selected Java test worktree")?;
                let source = TaskSourceKind::groovy_jvm_test(plan.worktree_id);
                let mut resolved = plan
                    .template
                    .resolve_task(&source.to_id_base(), task_context)
                    .context("The remote Java test Task could not be resolved")?;
                resolved.resolved.show_rerun = false;
                let handle = workspace.update_in(cx, |workspace, window, cx| {
                    workspace.schedule_resolved_task_with_structured_handle(
                        source, resolved, true, window, cx,
                    )
                })?;
                let registration = provider.update(cx, |provider, cx| {
                    provider.register_remote_run_handle(plan.clone(), handle.clone(), cx)
                });
                if let Err(error) = registration {
                    if let Err(cancel_error) = cx.update(|_, cx| handle.cancel(cx)) {
                        log::warn!(
                            "Failed to cancel an untracked remote Java test Task: {cancel_error:#}"
                        );
                    }
                    return Err(error);
                }
                registered = true;
                anyhow::Ok(())
            })();
            if let Err(error) = result {
                if !registered
                    && let Err(report_error) = provider.read_with(cx, |provider, cx| {
                        provider.fail_remote_action_plan(&plan, cx)
                    })
                {
                    log::warn!(
                        "Failed to report an unstarted remote Java test action: {report_error:#}"
                    );
                }
                Err::<(), _>(error).notify_workspace_async_err(workspace, &mut cx);
            }
            anyhow::Ok(())
        })
        .detach();
    Ok(())
}

#[cfg(all(feature = "rust-test-actions", feature = "jvm-test-actions"))]
struct CombinedTestExplorerDelegate;

#[cfg(all(feature = "rust-test-actions", feature = "jvm-test-actions"))]
impl CombinedTestExplorerDelegate {
    fn for_selection(
        selection: &TestExplorerSelection,
    ) -> anyhow::Result<&'static dyn TestExplorerActionDelegate> {
        static RUST: RustTestExplorerDelegate = RustTestExplorerDelegate;
        static JVM: JvmTestExplorerDelegate = JvmTestExplorerDelegate;
        match selection.provider_id.0.as_str() {
            RUST_TEST_PROVIDER_ID => Ok(&RUST),
            JVM_JUNIT_PROVIDER_ID => Ok(&JVM),
            _ => anyhow::bail!("The selected structured test provider is unavailable"),
        }
    }
}

#[cfg(all(feature = "rust-test-actions", feature = "jvm-test-actions"))]
impl TestExplorerActionDelegate for CombinedTestExplorerDelegate {
    fn discover(
        &self,
        project: &Entity<Project>,
        cx: &mut App,
    ) -> Vec<(
        StructuredProviderId,
        gpui::Task<anyhow::Result<StructuredProviderSnapshot>>,
    )> {
        let mut discoveries = RustTestExplorerDelegate.discover(project, cx);
        discoveries.extend(JvmTestExplorerDelegate.discover(project, cx));
        discoveries
    }

    fn run(
        &self,
        selection: &TestExplorerSelection,
        workspace: &WeakEntity<Workspace>,
        window: &mut Window,
        cx: &mut App,
    ) -> anyhow::Result<()> {
        Self::for_selection(selection)?.run(selection, workspace, window, cx)
    }

    fn cancel(
        &self,
        selection: &TestExplorerSelection,
        workspace: &WeakEntity<Workspace>,
        window: &mut Window,
        cx: &mut App,
    ) -> anyhow::Result<()> {
        Self::for_selection(selection)?.cancel(selection, workspace, window, cx)
    }

    fn debug(
        &self,
        selection: &TestExplorerSelection,
        workspace: &WeakEntity<Workspace>,
        window: &mut Window,
        cx: &mut App,
    ) -> anyhow::Result<()> {
        Self::for_selection(selection)?.debug(selection, workspace, window, cx)
    }

    fn rerun_failed(
        &self,
        selections: &[TestExplorerSelection],
        workspace: &WeakEntity<Workspace>,
        window: &mut Window,
        cx: &mut App,
    ) -> anyhow::Result<()> {
        anyhow::ensure!(
            selections.iter().all(|selection| matches!(
                selection.provider_id.0.as_str(),
                RUST_TEST_PROVIDER_ID | JVM_JUNIT_PROVIDER_ID
            )),
            "A failed selection belongs to an unavailable structured test provider"
        );
        let (rust, java): (Vec<_>, Vec<_>) = selections
            .iter()
            .cloned()
            .partition(|selection| selection.provider_id.0 == RUST_TEST_PROVIDER_ID);
        if !java.is_empty() {
            JvmTestExplorerDelegate.rerun_failed(&java, workspace, window, cx)?;
        }
        if !rust.is_empty() {
            RustTestExplorerDelegate.rerun_failed(&rust, workspace, window, cx)?;
        }
        Ok(())
    }

    fn reveal_terminal(
        &self,
        selection: &TestExplorerSelection,
        workspace: &WeakEntity<Workspace>,
        window: &mut Window,
        cx: &mut App,
    ) -> anyhow::Result<()> {
        Self::for_selection(selection)?.reveal_terminal(selection, workspace, window, cx)
    }
}

#[cfg(feature = "rust-test-actions")]
struct RustTestExplorerDelegate;

#[cfg(feature = "rust-test-actions")]
impl TestExplorerActionDelegate for RustTestExplorerDelegate {
    fn discover(
        &self,
        project: &Entity<Project>,
        cx: &mut App,
    ) -> Vec<(
        StructuredProviderId,
        gpui::Task<anyhow::Result<StructuredProviderSnapshot>>,
    )> {
        let store = project.read(cx).rust_test_provider_store().clone();
        vec![(
            StructuredProviderId(RUST_TEST_PROVIDER_ID.to_owned()),
            store.update(cx, |store, cx| store.refresh(cx)),
        )]
    }

    fn run(
        &self,
        selection: &TestExplorerSelection,
        workspace: &WeakEntity<Workspace>,
        window: &mut Window,
        cx: &mut App,
    ) -> anyhow::Result<()> {
        dispatch_rust_test_action(selection, RustTestAction::Run, workspace, window, cx)
    }

    fn debug(
        &self,
        selection: &TestExplorerSelection,
        workspace: &WeakEntity<Workspace>,
        window: &mut Window,
        cx: &mut App,
    ) -> anyhow::Result<()> {
        dispatch_rust_test_action(selection, RustTestAction::Debug, workspace, window, cx)
    }

    fn cancel(
        &self,
        selection: &TestExplorerSelection,
        workspace: &WeakEntity<Workspace>,
        _: &mut Window,
        cx: &mut App,
    ) -> anyhow::Result<()> {
        require_rust_provider(selection)?;
        let run_id = selection
            .run_id
            .as_ref()
            .context("The selected Rust test has no active run")?;
        let workspace = workspace
            .upgrade()
            .context("The workspace is no longer available")?;
        let store = workspace
            .read(cx)
            .project()
            .read(cx)
            .rust_test_provider_store()
            .clone();
        store.update(cx, |store, cx| store.cancel_run(run_id, cx))
    }

    fn rerun_failed(
        &self,
        selections: &[TestExplorerSelection],
        workspace: &WeakEntity<Workspace>,
        window: &mut Window,
        cx: &mut App,
    ) -> anyhow::Result<()> {
        let workspace_entity = workspace
            .upgrade()
            .context("The workspace is no longer available")?;
        let project = workspace_entity.read(cx).project().clone();
        let provider_store = project.read(cx).rust_test_provider_store().clone();
        let mut current = Vec::new();
        let mut skipped = 0usize;
        for selection in selections {
            require_rust_provider(selection)?;
            if rust_failed_selection_is_current(selection, &project, cx)
                && current.len() < MAX_RUST_TEST_RERUNS
            {
                let pending_operation = provider_store
                    .read(cx)
                    .action_experimental_requirements(
                        &selection.node_id,
                        selection.discovery_generation,
                        cx,
                    )
                    .and_then(|requirements| {
                        PendingExperimentalFeatureOperation::new(&requirements, cx)
                    });
                match pending_operation {
                    Ok(operation) => current.push((selection.clone(), operation)),
                    Err(_) => skipped = skipped.saturating_add(1),
                }
            } else {
                skipped = skipped.saturating_add(1);
            }
        }
        if current.is_empty() {
            anyhow::bail!("{}", rust_rerun_summary(0, skipped));
        }
        let task_contexts = workspace_entity.update(cx, |workspace, cx| {
            crate::task_contexts(workspace, window, cx)
        });
        let workspace = workspace_entity.downgrade();
        window
            .spawn(cx, async move |mut cx| {
                let task_contexts = task_contexts.await;
                let mut scheduled = 0usize;
                let mut skipped = skipped;
                let mut remaining = current.into_iter();
                while let Some((selection, pending_operation)) = remaining.next() {
                    let still_current =
                        cx.update(|_, cx| rust_test_case_is_current(&selection, &project, cx));
                    if !matches!(still_current, Ok(true)) {
                        skipped = skipped.saturating_add(1);
                        continue;
                    }
                    let plan = cx.update(|_, cx| {
                        pending_operation.check(cx)?;
                        Ok::<_, anyhow::Error>(provider_store.update(cx, |store, cx| {
                            store.plan_action(
                                &selection.node_id,
                                selection.discovery_generation,
                                RustTestAction::Run,
                                cx,
                            )
                        }))
                    });
                    let plan = match plan {
                        Ok(Ok(plan)) => plan.await,
                        Ok(Err(error)) => Err(error),
                        Err(error) => Err(error),
                    };
                    let handle = match plan {
                        Ok(plan) => schedule_rust_test_task_plan(
                            plan,
                            &pending_operation,
                            &task_contexts,
                            &provider_store,
                            &workspace,
                            &mut cx,
                        ),
                        Err(error) => Err(error),
                    };
                    let handle = match handle {
                        Ok(handle) => handle,
                        Err(error) => {
                            log::warn!("Failed to schedule a current Rust test rerun: {error:#}");
                            skipped = skipped.saturating_add(1);
                            continue;
                        }
                    };
                    let (terminal_sender, mut terminal_receiver) =
                        futures::channel::mpsc::unbounded();
                    let subscription = cx.update(|_, cx| {
                        handle.subscribe(cx, move |event, _| {
                            if event.state.is_terminal()
                                && terminal_sender.unbounded_send(event.state.clone()).is_err()
                            {
                                log::debug!("Rust test rerun completion receiver was dropped");
                            }
                        });
                    });
                    if let Err(error) = subscription {
                        log::warn!("Failed to observe a Rust test rerun: {error:#}");
                        skipped = skipped.saturating_add(1usize.saturating_add(remaining.count()));
                        break;
                    }
                    match terminal_receiver.next().await {
                        Some(task::StructuredTaskState::Completed { .. }) => {
                            scheduled = scheduled.saturating_add(1);
                        }
                        Some(task::StructuredTaskState::Cancelled { .. }) => {
                            skipped =
                                skipped.saturating_add(1usize.saturating_add(remaining.count()));
                            break;
                        }
                        Some(task::StructuredTaskState::SpawnError { .. }) | None => {
                            skipped = skipped.saturating_add(1);
                        }
                        Some(
                            task::StructuredTaskState::Queued
                            | task::StructuredTaskState::Running { .. },
                        ) => {
                            log::warn!("Rust test rerun observer received a nonterminal event");
                            skipped =
                                skipped.saturating_add(1usize.saturating_add(remaining.count()));
                            break;
                        }
                    }
                }
                if skipped > 0 || scheduled == 0 {
                    Err::<(), _>(anyhow::anyhow!(rust_rerun_summary(scheduled, skipped)))
                        .notify_workspace_async_err(workspace, &mut cx);
                }
                anyhow::Ok(())
            })
            .detach();
        Ok(())
    }

    fn reveal_terminal(
        &self,
        selection: &TestExplorerSelection,
        workspace: &WeakEntity<Workspace>,
        window: &mut Window,
        cx: &mut App,
    ) -> anyhow::Result<()> {
        require_rust_provider(selection)?;
        let run_id = selection
            .run_id
            .as_ref()
            .context("The selected Rust test has no retained run")?;
        let workspace = workspace
            .upgrade()
            .context("The workspace is no longer available")?;
        let store = workspace
            .read(cx)
            .project()
            .read(cx)
            .rust_test_provider_store()
            .clone();
        store.update(cx, |store, cx| {
            store.reveal_run_terminal(run_id, window, cx)
        })
    }
}

#[cfg(feature = "rust-test-actions")]
fn require_rust_provider(selection: &TestExplorerSelection) -> anyhow::Result<()> {
    anyhow::ensure!(
        selection.provider_id.0 == RUST_TEST_PROVIDER_ID,
        "The selected structured provider is not a Rust test provider"
    );
    Ok(())
}

#[cfg(feature = "rust-test-actions")]
fn rust_test_case_is_current(
    selection: &TestExplorerSelection,
    project: &Entity<Project>,
    cx: &App,
) -> bool {
    let structured_store = project.read(cx).structured_execution_store().clone();
    let structured_store = structured_store.read(cx);
    let Some(snapshot) = structured_store.state().provider(&selection.provider_id) else {
        return false;
    };
    snapshot.discovery_generation == selection.discovery_generation
        && matches!(
            snapshot.status,
            StructuredProviderStatus::Current | StructuredProviderStatus::Partial
        )
        && snapshot
            .nodes
            .iter()
            .any(|node| node.id == selection.node_id && node.kind == StructuredNodeKind::Case)
}

#[cfg(feature = "rust-test-actions")]
fn rust_failed_selection_is_current(
    selection: &TestExplorerSelection,
    project: &Entity<Project>,
    cx: &App,
) -> bool {
    if !rust_test_case_is_current(selection, project, cx) {
        return false;
    }
    let Some(run_id) = selection.run_id.as_ref() else {
        return false;
    };
    let structured_store = project.read(cx).structured_execution_store().clone();
    let structured_store = structured_store.read(cx);
    let Some(snapshot) = structured_store.state().provider(&selection.provider_id) else {
        return false;
    };
    let run = snapshot
        .current_run
        .as_ref()
        .or(snapshot.last_complete_run.as_ref());
    run.is_some_and(|run| {
        &run.id == run_id
            && run
                .events
                .iter()
                .rev()
                .find(|event| event.node_id == selection.node_id)
                .is_some_and(|event| event.state == StructuredNodeState::Failed)
    })
}

#[cfg(feature = "rust-test-actions")]
fn rust_rerun_summary(scheduled: usize, skipped: usize) -> String {
    format!(
        "Ran {scheduled} failed Rust tests; skipped {skipped} unavailable, removed, stale, or over-limit selections"
    )
}

#[cfg(feature = "rust-test-actions")]
fn schedule_rust_test_task_plan(
    plan: RustTestActionPlan,
    pending_operation: &PendingExperimentalFeatureOperation,
    task_contexts: &project::TaskContexts,
    provider_store: &Entity<project::rust_test_provider::RustTestProviderStore>,
    workspace: &WeakEntity<Workspace>,
    cx: &mut AsyncWindowContext,
) -> anyhow::Result<task::StructuredTaskHandle> {
    cx.update(|_, cx| {
        pending_operation.check_requirements(plan.required_experimental_features(), cx)
    })??;
    let RustTestActionPlan::Task {
        run_id,
        discovery_generation,
        scope_node_ids,
        worktree_id,
        template,
    } = plan
    else {
        anyhow::bail!("A Rust test rerun must resolve to a Task")
    };
    let task_context = task_contexts
        .task_context_for_worktree_id(worktree_id)
        .context("No task context exists for the selected Rust test worktree")?;
    let source = TaskSourceKind::Language {
        name: "Rust Tests".into(),
    };
    let resolved = resolve_rust_test_task(&template, &source, task_context)?;
    let handle = workspace.update_in(cx, |workspace, window, cx| {
        workspace.schedule_resolved_task_with_structured_handle(source, resolved, false, window, cx)
    })?;
    let registered = provider_store.update(cx, |store, cx| {
        store.register_run_handle(
            run_id,
            discovery_generation,
            scope_node_ids,
            handle.clone(),
            cx,
        )
    });
    if let Err(error) = registered {
        if let Err(cancel_error) = cx.update(|_, cx| handle.cancel(cx)) {
            log::warn!("Failed to cancel an untracked Rust test task: {cancel_error:#}");
        }
        return Err(error);
    }
    Ok(handle)
}

#[cfg(feature = "rust-test-actions")]
fn resolve_rust_test_task(
    template: &task::TaskTemplate,
    source: &TaskSourceKind,
    task_context: &task::TaskContext,
) -> anyhow::Result<task::ResolvedTask> {
    let mut template = template.clone();
    // The label is provider data, while the cwd still resolves its Task context.
    template.label = template.label.replace('$', "$$");
    template
        .resolve_task_with_literal_arguments(&source.to_id_base(), task_context)
        .context("The Rust test task could not be resolved")
}

#[cfg(feature = "rust-test-actions")]
async fn await_rust_test_dispatch(
    plan: gpui::Task<anyhow::Result<RustTestActionPlan>>,
    task_contexts: gpui::Task<project::TaskContexts>,
    pending_operation: &PendingExperimentalFeatureOperation,
    cx: &mut AsyncWindowContext,
) -> anyhow::Result<(RustTestActionPlan, project::TaskContexts)> {
    let task_contexts = task_contexts.await;
    let plan = plan.await?;
    cx.update(|_, cx| {
        pending_operation.check_requirements(plan.required_experimental_features(), cx)
    })??;
    Ok((plan, task_contexts))
}

#[cfg(feature = "rust-test-actions")]
fn dispatch_rust_test_action(
    selection: &TestExplorerSelection,
    action: RustTestAction,
    workspace: &WeakEntity<Workspace>,
    window: &mut Window,
    cx: &mut App,
) -> anyhow::Result<()> {
    use task::SharedTaskContext;

    require_rust_provider(selection)?;
    let workspace_entity = workspace
        .upgrade()
        .context("The workspace is no longer available")?;
    let project = workspace_entity.read(cx).project().clone();
    let provider_store = project.read(cx).rust_test_provider_store().clone();
    let requirements = provider_store.read(cx).action_experimental_requirements(
        &selection.node_id,
        selection.discovery_generation,
        cx,
    )?;
    let pending_operation = PendingExperimentalFeatureOperation::new(&requirements, cx)?;
    let plan = provider_store.update(cx, |store, cx| {
        store.plan_action(
            &selection.node_id,
            selection.discovery_generation,
            action,
            cx,
        )
    });
    let task_contexts = workspace_entity.update(cx, |workspace, cx| {
        crate::task_contexts(workspace, window, cx)
    });
    let workspace = workspace_entity.downgrade();
    window
        .spawn(cx, async move |mut cx| {
            let prepared =
                await_rust_test_dispatch(plan, task_contexts, &pending_operation, &mut cx).await;
            let Some((plan, task_contexts)) =
                prepared.notify_workspace_async_err(workspace.clone(), &mut cx)
            else {
                return anyhow::Ok(());
            };
            let result = match plan {
                plan @ RustTestActionPlan::Task { .. } => schedule_rust_test_task_plan(
                    plan,
                    &pending_operation,
                    &task_contexts,
                    &provider_store,
                    &workspace,
                    &mut cx,
                )
                .map(|_| ()),
                RustTestActionPlan::Debug {
                    worktree_id,
                    scenario,
                } => {
                    let task_context = task_contexts
                        .task_context_for_worktree_id(worktree_id)
                        .cloned()
                        .context("No task context exists for the selected Rust test worktree")?;
                    workspace.update_in(cx, |workspace, window, cx| {
                        workspace.start_debug_session(
                            scenario,
                            SharedTaskContext::from(task_context),
                            None,
                            Some(worktree_id),
                            window,
                            cx,
                        )
                    })?;
                    anyhow::Ok(())
                }
            };
            if let Err(error) = result {
                Err::<(), _>(error).notify_workspace_async_err(workspace, &mut cx);
            }
            anyhow::Ok(())
        })
        .detach();
    Ok(())
}

#[derive(Clone, Debug)]
pub struct TestExplorerProjection {
    pub snapshot: LanguageToolSnapshot,
    pub summary: StructuredExecutionSummary,
    aggregate_run_count: usize,
    navigation: HashMap<LanguageToolNodeId, ProjectPath>,
    selections: HashMap<LanguageToolNodeId, TestExplorerSelection>,
    node_kinds: HashMap<LanguageToolNodeId, StructuredNodeKind>,
    debug_disabled_reasons: HashMap<LanguageToolNodeId, String>,
    experimental_features: HashMap<LanguageToolNodeId, (String, Option<bool>)>,
    failed: Vec<TestExplorerSelection>,
}

impl Default for TestExplorerProjection {
    fn default() -> Self {
        Self {
            snapshot: LanguageToolSnapshot {
                roots: Vec::new(),
                status: LanguageToolProviderStatus::Empty(
                    "No structured test providers are available".to_string(),
                ),
            },
            summary: StructuredExecutionSummary::default(),
            aggregate_run_count: 0,
            navigation: HashMap::new(),
            selections: HashMap::new(),
            node_kinds: HashMap::new(),
            debug_disabled_reasons: HashMap::new(),
            experimental_features: HashMap::new(),
            failed: Vec::new(),
        }
    }
}

impl TestExplorerProjection {
    fn summary_label(&self) -> String {
        if self.aggregate_run_count > 0 && self.summary.total == 0 {
            return "Per-test results unavailable; task status is shown on provider rows".into();
        }
        let summary = format!(
            "{} total · {} passed · {} failed · {} skipped",
            self.summary.total, self.summary.passed, self.summary.failed, self.summary.skipped
        );
        if self.aggregate_run_count == 0 {
            summary
        } else {
            format!("{summary}; some tasks have no per-test results")
        }
    }

    pub fn navigation(&self, id: &LanguageToolNodeId) -> Option<&ProjectPath> {
        self.navigation.get(id)
    }

    pub fn selection(&self, id: &LanguageToolNodeId) -> Option<&TestExplorerSelection> {
        self.selections.get(id)
    }

    fn can_run(&self, id: &LanguageToolNodeId) -> bool {
        let Some(kind) = self.node_kinds.get(id) else {
            return false;
        };
        #[cfg(feature = "jvm-test-actions")]
        if self
            .selections
            .get(id)
            .is_some_and(|selection| selection.provider_id.0 == JVM_JUNIT_PROVIDER_ID)
        {
            return *kind == StructuredNodeKind::Case;
        }
        *kind != StructuredNodeKind::Provider
    }

    fn debug_disabled_reason(&self, id: &LanguageToolNodeId) -> Option<&str> {
        if self.node_kinds.get(id) != Some(&StructuredNodeKind::Case) {
            return Some("Select an individual test case to debug");
        }
        self.debug_disabled_reasons.get(id).map(String::as_str)
    }

    fn can_debug(&self, id: &LanguageToolNodeId) -> bool {
        self.debug_disabled_reason(id).is_none()
    }

    fn experimental_disabled_reason(&self, id: &LanguageToolNodeId, cx: &App) -> Option<String> {
        self.experimental_features.get(id).and_then(|(feature, host_enabled)| {
            if *host_enabled != Some(true) {
                Some("Experimental test action is unavailable; enable it on the project host and refresh Tests".to_string())
            } else {
                (!cx.has_flag_by_name(feature)).then(|| {
                    format!("Experimental test action is disabled; enable {feature} in Feature Flags")
                })
            }
        })
    }

    fn can_run_with_flags(&self, id: &LanguageToolNodeId, cx: &App) -> bool {
        self.can_run(id) && self.experimental_disabled_reason(id, cx).is_none()
    }

    fn debug_disabled_reason_with_flags(
        &self,
        id: &LanguageToolNodeId,
        cx: &App,
    ) -> Option<String> {
        self.experimental_disabled_reason(id, cx)
            .or_else(|| self.debug_disabled_reason(id).map(str::to_owned))
    }

    fn can_debug_with_flags(&self, id: &LanguageToolNodeId, cx: &App) -> bool {
        self.can_debug(id) && self.experimental_disabled_reason(id, cx).is_none()
    }

    fn enabled_failed(&self, cx: &App) -> Vec<TestExplorerSelection> {
        self.failed
            .iter()
            .filter(|selection| {
                self.can_run_with_flags(&tree_id(&selection.provider_id, &selection.node_id), cx)
            })
            .cloned()
            .collect()
    }

    pub fn failed(&self) -> &[TestExplorerSelection] {
        &self.failed
    }
}

pub fn project_test_explorer(
    providers: &[StructuredProviderSnapshot],
    filter: &TestExplorerFilter,
) -> TestExplorerProjection {
    if providers.is_empty() {
        return TestExplorerProjection::default();
    }
    let query = filter.normalized_query();
    let mut providers = providers.to_vec();
    providers.sort_by(|left, right| left.provider_id.cmp(&right.provider_id));
    let mut projection = TestExplorerProjection {
        snapshot: LanguageToolSnapshot {
            roots: Vec::new(),
            status: combined_provider_status(&providers),
        },
        ..TestExplorerProjection::default()
    };

    for provider in &providers {
        let run = provider
            .current_run
            .as_ref()
            .or(provider.last_complete_run.as_ref());
        if let Some(run) = run {
            add_summary(&mut projection.summary, &run.summary);
            if run.events.is_empty() && run.summary.total == 0 {
                projection.aggregate_run_count = projection.aggregate_run_count.saturating_add(1);
            }
        }
        let events = run
            .map(|run| {
                run.events
                    .iter()
                    .map(|event| (event.node_id.clone(), event))
                    .collect::<HashMap<_, _>>()
            })
            .unwrap_or_default();
        let mut children = HashMap::<Option<StructuredNodeId>, Vec<StructuredNode>>::new();
        for node in &provider.nodes {
            children
                .entry(node.parent_id.clone())
                .or_default()
                .push(node.clone());
        }
        for nodes in children.values_mut() {
            nodes.sort_by(|left, right| left.id.cmp(&right.id));
        }
        let mut visiting = HashSet::new();
        let roots = children.get(&None).cloned().unwrap_or_default();
        for node in roots {
            if let Some(node) = project_node(
                provider,
                run,
                &node,
                &children,
                &events,
                filter,
                &query,
                false,
                0,
                &mut visiting,
                &mut projection,
            ) {
                projection.snapshot.roots.push(node);
            }
        }
    }

    if projection.snapshot.roots.is_empty()
        && matches!(
            projection.snapshot.status,
            LanguageToolProviderStatus::Current
        )
    {
        projection.snapshot.status = LanguageToolProviderStatus::Empty(
            if query.is_empty() && filter.status == TestStatusFilter::All {
                "No tests were discovered".to_string()
            } else {
                "No tests match the current filters".to_string()
            },
        );
    }
    projection
}

#[allow(clippy::too_many_arguments)]
fn project_node(
    provider: &StructuredProviderSnapshot,
    run: Option<&StructuredRun>,
    node: &StructuredNode,
    children: &HashMap<Option<StructuredNodeId>, Vec<StructuredNode>>,
    events: &HashMap<StructuredNodeId, &StructuredExecutionEvent>,
    filter: &TestExplorerFilter,
    query: &str,
    ancestor_query_match: bool,
    depth: usize,
    visiting: &mut HashSet<StructuredNodeId>,
    projection: &mut TestExplorerProjection,
) -> Option<LanguageToolNode> {
    if depth >= MAX_TREE_DEPTH || !visiting.insert(node.id.clone()) {
        return None;
    }
    let event = events.get(&node.id).copied();
    let aggregate_run = run.filter(|run| {
        node.kind == StructuredNodeKind::Provider && run.events.is_empty() && run.summary.total == 0
    });
    let text_matches =
        ancestor_query_match || query.is_empty() || node.label.to_lowercase().contains(query);
    let mut projected_children = Vec::new();
    if let Some(node_children) = children.get(&Some(node.id.clone())) {
        for child in node_children {
            if let Some(child) = project_node(
                provider,
                run,
                child,
                children,
                events,
                filter,
                query,
                text_matches,
                depth.saturating_add(1),
                visiting,
                projection,
            ) {
                projected_children.push(child);
            }
        }
    }
    visiting.remove(&node.id);

    let aggregate_state = aggregate_run.and_then(|run| match run.phase {
        StructuredRunPhase::Queued => Some(StructuredNodeState::Queued),
        StructuredRunPhase::Running => Some(StructuredNodeState::Running),
        StructuredRunPhase::Cancelled => Some(StructuredNodeState::Cancelled),
        StructuredRunPhase::SpawnError => Some(StructuredNodeState::Failed),
        // Completion does not encode a per-test outcome or make a diagnostic a result.
        StructuredRunPhase::Completed => None,
    });
    let status_matches = filter
        .status
        .matches(event.map(|event| event.state).or(aggregate_state));
    if (!text_matches || !status_matches) && projected_children.is_empty() {
        return None;
    }
    let tree_id = tree_id(&provider.provider_id, &node.id);
    let navigation = event
        .and_then(|event| event.location.clone())
        .or_else(|| node.path.clone());
    if let Some(path) = navigation {
        projection.navigation.insert(tree_id.clone(), path);
    }
    let navigable = projection.navigation.contains_key(&tree_id);
    let selection = TestExplorerSelection {
        provider_id: provider.provider_id.clone(),
        discovery_generation: provider.discovery_generation,
        node_id: node.id.clone(),
        run_id: run.map(|run| run.id.clone()),
    };
    if event.is_some_and(|event| event.state == StructuredNodeState::Failed) {
        projection.failed.push(selection.clone());
    }
    projection.selections.insert(tree_id.clone(), selection);
    projection.node_kinds.insert(tree_id.clone(), node.kind);
    if let Some(feature) = &node.experimental_feature {
        projection.experimental_features.insert(
            tree_id.clone(),
            (feature.clone(), node.experimental_feature_enabled),
        );
    }
    if node.kind == StructuredNodeKind::Case {
        let reason = if !matches!(
            provider.status,
            StructuredProviderStatus::Current | StructuredProviderStatus::Partial
        ) {
            Some("Refresh test discovery before debugging".to_string())
        } else {
            match &node.debug_capability {
                Some(StructuredDebugCapability::Supported) => None,
                Some(StructuredDebugCapability::Unsupported(reason)) => Some(reason.clone()),
                None => Some("This test provider has not reported Debug support".to_string()),
            }
        };
        if let Some(reason) = reason {
            projection
                .debug_disabled_reasons
                .insert(tree_id.clone(), reason);
        }
    }

    let aggregate_label = aggregate_run.map(aggregate_run_label);
    let accessible_label = aggregate_label
        .as_ref()
        .map(|label| format!("{}, {label}", node.label))
        .unwrap_or_else(|| accessibility_label(node, event));
    Some(LanguageToolNode {
        id: tree_id,
        label: node.label.clone(),
        secondary_label: aggregate_label.or_else(|| event.map(event_label)),
        icon: Some(node_icon(
            node.kind,
            event.map(|event| event.state).or(aggregate_state),
        )),
        accessibility_label: accessible_label,
        children: projected_children,
        enabled: navigable,
        activation_label: navigable.then(|| "Open test source or failure location".to_string()),
    })
}

fn aggregate_run_label(run: &StructuredRun) -> String {
    let phase = match run.phase {
        StructuredRunPhase::Queued => "task queued",
        StructuredRunPhase::Running => "task running",
        StructuredRunPhase::Completed => "task completed",
        StructuredRunPhase::SpawnError => "task failed to start",
        StructuredRunPhase::Cancelled => "task cancelled",
    };
    let diagnostic = run.diagnostic.as_deref().filter(|value| !value.is_empty());
    match diagnostic {
        Some(diagnostic) => format!("{phase}; {diagnostic}; per-test results unavailable"),
        None => format!("{phase}; per-test results unavailable"),
    }
}

fn combined_provider_status(
    providers: &[StructuredProviderSnapshot],
) -> LanguageToolProviderStatus {
    let selected = providers
        .iter()
        .min_by_key(|provider| provider_status_priority(provider.status));
    let Some(provider) = selected else {
        return LanguageToolProviderStatus::Empty(
            "No structured test providers are available".to_string(),
        );
    };
    let diagnostic = || {
        provider
            .diagnostic
            .clone()
            .unwrap_or_else(|| provider_status_default_message(provider.status).to_string())
    };
    match provider.status {
        StructuredProviderStatus::Loading => LanguageToolProviderStatus::Loading,
        StructuredProviderStatus::Current => LanguageToolProviderStatus::Current,
        StructuredProviderStatus::Empty => LanguageToolProviderStatus::Empty(diagnostic()),
        StructuredProviderStatus::Partial => LanguageToolProviderStatus::Partial(diagnostic()),
        StructuredProviderStatus::Stale => LanguageToolProviderStatus::Stale(diagnostic()),
        StructuredProviderStatus::Error => LanguageToolProviderStatus::Error(diagnostic()),
        StructuredProviderStatus::Restricted => {
            LanguageToolProviderStatus::Restricted(diagnostic())
        }
        StructuredProviderStatus::Disconnected => {
            LanguageToolProviderStatus::Disconnected(diagnostic())
        }
        StructuredProviderStatus::Mismatch => LanguageToolProviderStatus::Mismatch(diagnostic()),
    }
}

fn provider_status_priority(status: StructuredProviderStatus) -> u8 {
    match status {
        StructuredProviderStatus::Mismatch => 0,
        StructuredProviderStatus::Restricted => 1,
        StructuredProviderStatus::Disconnected => 2,
        StructuredProviderStatus::Error => 3,
        StructuredProviderStatus::Stale => 4,
        StructuredProviderStatus::Partial => 5,
        StructuredProviderStatus::Loading => 6,
        StructuredProviderStatus::Empty => 7,
        StructuredProviderStatus::Current => 8,
    }
}

fn provider_status_default_message(status: StructuredProviderStatus) -> &'static str {
    match status {
        StructuredProviderStatus::Loading => "Test discovery is loading",
        StructuredProviderStatus::Current => "Test discovery is current",
        StructuredProviderStatus::Empty => "No tests were discovered",
        StructuredProviderStatus::Partial => "Some tests could not be discovered",
        StructuredProviderStatus::Stale => "Test discovery is stale; refresh to retry",
        StructuredProviderStatus::Error => "Test discovery failed; refresh to retry",
        StructuredProviderStatus::Restricted => "Trust this project to discover and run tests",
        StructuredProviderStatus::Disconnected => "Reconnect to the project host to load tests",
        StructuredProviderStatus::Mismatch => {
            "The project host does not support this structured test protocol"
        }
    }
}

fn tree_id(provider_id: &StructuredProviderId, node_id: &StructuredNodeId) -> LanguageToolNodeId {
    LanguageToolNodeId(format!(
        "provider:{}:{}:node:{}:{}",
        provider_id.0.len(),
        provider_id.0,
        node_id.0.len(),
        node_id.0
    ))
}

fn node_icon(kind: StructuredNodeKind, state: Option<StructuredNodeState>) -> IconName {
    match state {
        Some(StructuredNodeState::Passed) => IconName::Check,
        Some(StructuredNodeState::Failed) => IconName::XCircle,
        Some(StructuredNodeState::Skipped) => IconName::Circle,
        Some(StructuredNodeState::Queued) => IconName::Clock,
        Some(StructuredNodeState::Running) => IconName::LoadCircle,
        Some(StructuredNodeState::Cancelled) => IconName::Stop,
        None => match kind {
            StructuredNodeKind::Provider => IconName::ListTree,
            StructuredNodeKind::Suite => IconName::Folder,
            StructuredNodeKind::Group => IconName::ListTree,
            StructuredNodeKind::Case => IconName::Circle,
        },
    }
}

fn event_label(event: &StructuredExecutionEvent) -> String {
    let status = match event.state {
        StructuredNodeState::Queued => "queued",
        StructuredNodeState::Running => "running",
        StructuredNodeState::Passed => "passed",
        StructuredNodeState::Failed => "failed",
        StructuredNodeState::Skipped => "skipped",
        StructuredNodeState::Cancelled => "cancelled",
    };
    event
        .duration_millis
        .map(|duration| format!("{status} · {duration} ms"))
        .unwrap_or_else(|| status.to_string())
}

fn accessibility_label(node: &StructuredNode, event: Option<&StructuredExecutionEvent>) -> String {
    event
        .map(|event| format!("{}, {}", node.label, event_label(event)))
        .unwrap_or_else(|| node.label.clone())
}

fn add_summary(target: &mut StructuredExecutionSummary, source: &StructuredExecutionSummary) {
    target.total = target.total.saturating_add(source.total);
    target.queued = target.queued.saturating_add(source.queued);
    target.running = target.running.saturating_add(source.running);
    target.passed = target.passed.saturating_add(source.passed);
    target.failed = target.failed.saturating_add(source.failed);
    target.skipped = target.skipped.saturating_add(source.skipped);
    target.cancelled = target.cancelled.saturating_add(source.cancelled);
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct TestsPanelSettings {
    button: bool,
    default_width: Pixels,
    dock: DockSide,
    starts_open: bool,
}

impl Settings for TestsPanelSettings {
    fn from_settings(content: &settings::SettingsContent) -> Self {
        let settings = content.tests_panel.as_ref();
        Self {
            button: settings
                .and_then(|settings| settings.button)
                .unwrap_or(true),
            default_width: gpui::px(
                settings
                    .and_then(|settings| settings.default_width)
                    .unwrap_or(300.),
            ),
            dock: settings
                .and_then(|settings| settings.dock)
                .unwrap_or(DockSide::Left),
            starts_open: settings
                .and_then(|settings| settings.starts_open)
                .unwrap_or(false),
        }
    }
}

#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
struct PersistedTestsPanelState {
    version: u32,
    query: String,
    status: TestStatusFilter,
    selected_provider: Option<String>,
    selected_node: Option<String>,
}

impl PersistedTestsPanelState {
    fn sanitized(mut self) -> Self {
        self.version = TESTS_PANEL_STATE_VERSION;
        self.query = bounded_filter(&self.query);
        self.selected_provider = self.selected_provider.map(|value| bounded_filter(&value));
        self.selected_node = self.selected_node.map(|value| bounded_filter(&value));
        self
    }
}

fn bounded_filter(value: &str) -> String {
    if value.len() <= MAX_FILTER_BYTES {
        return value.to_string();
    }
    let mut end = MAX_FILTER_BYTES;
    while end > 0 && !value.is_char_boundary(end) {
        end -= 1;
    }
    value[..end].to_string()
}

fn persist_state(key: String, state: PersistedTestsPanelState, cx: &App) -> gpui::Task<()> {
    let key_value_store = KeyValueStore::global(cx);
    cx.background_spawn(async move {
        let result = async {
            key_value_store
                .write_kvp(key, serde_json::to_string(&state.sanitized())?)
                .await?;
            anyhow::Ok(())
        }
        .await;
        if let Err(error) = result {
            log::warn!("failed to persist Tests panel state: {error:#}");
        }
    })
}

#[cfg(test)]
#[derive(Default)]
struct TestsProjectionTimings {
    refresh: std::time::Duration,
    admission: std::time::Duration,
    application: std::time::Duration,
    background_nanoseconds: Arc<std::sync::atomic::AtomicU64>,
    background_gate: Option<futures::channel::oneshot::Receiver<()>>,
}

pub struct TestsPanel {
    project: Entity<Project>,
    workspace: WeakEntity<Workspace>,
    focus_handle: FocusHandle,
    query_input: Entity<InputField>,
    active: bool,
    host: LanguageToolTreeHost,
    filter: TestExplorerFilter,
    projection: TestExplorerProjection,
    snapshots: Arc<[StructuredProviderSnapshot]>,
    projection_task: gpui::Task<()>,
    projection_generation: u64,
    #[cfg(test)]
    projection_timings: TestsProjectionTimings,
    refresh_pending: bool,
    refresh_status: Option<LanguageToolProviderStatus>,
    provider_ids: Vec<StructuredProviderId>,
    restored_selection: Option<(StructuredProviderId, StructuredNodeId)>,
    delegate: Arc<dyn TestExplorerActionDelegate>,
    state_key: Option<String>,
    serialization_task: gpui::Task<()>,
    #[cfg(feature = "jvm-test-actions")]
    _java_language_task: gpui::Task<()>,
    #[cfg(feature = "jvm-test-actions")]
    java_discovery_pending: bool,
    action_notice: Option<String>,
    context_menu: Option<(Entity<ContextMenu>, Point<Pixels>, Subscription)>,
    _subscriptions: Vec<Subscription>,
}

impl TestsPanel {
    pub async fn load(
        workspace: WeakEntity<Workspace>,
        mut cx: AsyncWindowContext,
    ) -> anyhow::Result<Entity<Self>> {
        let workspace_handle = workspace.clone();
        workspace.update_in(&mut cx, |workspace, window, cx| {
            let project = workspace.project().clone();
            let state_key = workspace_scoped_state_key(
                TESTS_PANEL_STATE_KEY,
                workspace.database_id(),
                workspace.session_id().as_deref(),
            );
            let restored = state_key
                .as_ref()
                .and_then(|key| match KeyValueStore::global(cx).read_kvp(key) {
                    Ok(value) => value,
                    Err(error) => {
                        log::warn!("failed to restore Tests panel state: {error:#}");
                        None
                    }
                })
                .and_then(
                    |value| match serde_json::from_str::<PersistedTestsPanelState>(&value) {
                        Ok(state) if state.version == TESTS_PANEL_STATE_VERSION => {
                            Some(state.sanitized())
                        }
                        Ok(_) => None,
                        Err(error) => {
                            log::warn!("failed to parse Tests panel state: {error:#}");
                            None
                        }
                    },
                )
                .unwrap_or_default();
            let query_input = cx.new(|cx| {
                InputField::new(window, cx, "Filter tests…")
                    .start_icon(IconName::MagnifyingGlass)
                    .tab_index(0)
            });
            let query_editor = query_input.read(cx).editor().clone();
            query_editor.set_text(&restored.query, window, cx);
            cx.new(|cx: &mut Context<TestsPanel>| {
                let store = project.read(cx).structured_execution_store().clone();
                #[cfg(feature = "jvm-test-actions")]
                let lsp_store = project.read(cx).lsp_store();
                #[cfg(feature = "jvm-test-actions")]
                let java_language_task = {
                    let languages = project.read(cx).languages().clone();
                    let mut changes = languages.subscribe();
                    cx.spawn(async move |this, cx| {
                        loop {
                            if languages.language_for_name("Java").await.is_ok() {
                                if let Err(error) = this.update(cx, |panel, cx| {
                                    panel.request_java_discovery(false, cx);
                                }) {
                                    log::debug!("Tests panel was dropped: {error:#}");
                                }
                                break;
                            }
                            if changes.next().await.is_none() {
                                break;
                            }
                        }
                    })
                };
                let flag_subscription = cx.observe_global::<FeatureFlagStore>(|panel, cx| {
                    panel.context_menu = None;
                    cx.notify();
                });
                let store_subscription = cx
                    .subscribe(&store, |panel, _, _: &StructuredExecutionStoreEvent, cx| {
                        panel.invalidate(cx)
                    });
                #[cfg(feature = "jvm-test-actions")]
                let lsp_subscription =
                    cx.subscribe(&lsp_store, |panel, _, event: &LspStoreEvent, cx| {
                        if let LspStoreEvent::LanguageServerAdded(_, name, _) = event
                            && name.0.as_ref() == "jdtls"
                        {
                            panel.request_java_discovery(true, cx);
                        }
                    });
                let panel = cx.weak_entity();
                let query_editor = query_input.read(cx).editor().clone();
                let query_subscription = query_editor.subscribe(
                    Box::new(move |event, _, cx| {
                        if event == ErasedEditorEvent::BufferEdited {
                            if let Err(error) = panel.update(cx, |panel, cx| {
                                panel.filter.query =
                                    bounded_filter(&panel.query_input.read(cx).text(cx));
                                panel.rebuild_projection(cx);
                                panel.persist(cx);
                            }) {
                                log::debug!("Tests panel was dropped: {error:#}");
                            }
                        }
                    }),
                    window,
                    cx,
                );
                let provider_ids = store
                    .read(cx)
                    .state()
                    .providers()
                    .map(|provider| provider.provider_id.clone())
                    .collect();
                TestsPanel {
                    project,
                    workspace: workspace_handle,
                    focus_handle: cx.focus_handle(),
                    query_input,
                    active: false,
                    host: LanguageToolTreeHost::default(),
                    filter: TestExplorerFilter {
                        query: restored.query,
                        status: restored.status,
                    },
                    projection: TestExplorerProjection::default(),
                    snapshots: Arc::default(),
                    projection_task: gpui::Task::ready(()),
                    projection_generation: 0,
                    #[cfg(test)]
                    projection_timings: TestsProjectionTimings::default(),
                    refresh_pending: false,
                    refresh_status: None,
                    provider_ids,
                    restored_selection: restored.selected_provider.zip(restored.selected_node).map(
                        |(provider, node)| (StructuredProviderId(provider), StructuredNodeId(node)),
                    ),
                    delegate: default_test_explorer_delegate(),
                    state_key,
                    serialization_task: gpui::Task::ready(()),
                    #[cfg(feature = "jvm-test-actions")]
                    _java_language_task: java_language_task,
                    #[cfg(feature = "jvm-test-actions")]
                    java_discovery_pending: false,
                    action_notice: None,
                    context_menu: None,
                    _subscriptions: vec![
                        store_subscription,
                        flag_subscription,
                        query_subscription,
                        #[cfg(feature = "jvm-test-actions")]
                        lsp_subscription,
                    ],
                }
            })
        })
    }

    pub fn set_delegate(&mut self, delegate: Arc<dyn TestExplorerActionDelegate>) {
        self.delegate = delegate;
    }

    pub fn set_provider_ids(
        &mut self,
        provider_ids: impl IntoIterator<Item = StructuredProviderId>,
        cx: &mut Context<Self>,
    ) {
        self.provider_ids = provider_ids.into_iter().collect();
        self.provider_ids.sort();
        self.provider_ids.dedup();
        self.refresh(cx);
    }

    pub fn toggle_focus(
        workspace: &mut Workspace,
        _: &ToggleTestsPanel,
        window: &mut Window,
        cx: &mut Context<Workspace>,
    ) {
        workspace.toggle_panel_focus::<Self>(window, cx);
    }

    #[cfg(feature = "jvm-test-actions")]
    fn request_java_discovery(&mut self, force: bool, cx: &mut Context<Self>) {
        let store = self.project.read(cx).structured_execution_store().clone();
        let has_java_cases = |snapshot: &StructuredProviderSnapshot| {
            snapshot.provider_id.0 == JVM_JUNIT_PROVIDER_ID
                && snapshot
                    .nodes
                    .iter()
                    .any(|node| node.kind == StructuredNodeKind::Case)
        };
        if !force
            && (self.snapshots.iter().any(has_java_cases)
                || store.read(cx).state().providers().any(has_java_cases))
        {
            return;
        }
        let has_active_run = self.snapshots.iter().any(|snapshot| {
            snapshot
                .current_run
                .as_ref()
                .is_some_and(|run| !run.phase.is_terminal())
        }) || store.read(cx).state().providers().any(|snapshot| {
            snapshot
                .current_run
                .as_ref()
                .is_some_and(|run| !run.phase.is_terminal())
        });
        if has_active_run {
            self.java_discovery_pending = true;
        } else if self.active {
            self.discover(cx);
        } else {
            self.host.mark_dirty();
        }
    }

    fn invalidate(&mut self, cx: &mut Context<Self>) {
        #[cfg(feature = "jvm-test-actions")]
        if self.java_discovery_pending {
            let store = self.project.read(cx).structured_execution_store().clone();
            let has_active_run = store.read(cx).state().providers().any(|snapshot| {
                snapshot
                    .current_run
                    .as_ref()
                    .is_some_and(|run| !run.phase.is_terminal())
            });
            if !has_active_run {
                self.java_discovery_pending = false;
                let panel = cx.weak_entity();
                cx.defer(move |cx| {
                    if let Err(error) = panel.update(cx, |panel, cx| {
                        if panel.active {
                            panel.discover(cx);
                        } else {
                            panel.host.mark_dirty();
                        }
                    }) {
                        log::debug!("Tests panel was dropped: {error:#}");
                    }
                });
                return;
            }
        }
        self.cancel_projection();
        if !self.active || self.refresh_pending {
            self.host.mark_dirty();
        } else {
            self.refresh(cx);
        }
    }

    fn refresh(&mut self, cx: &mut Context<Self>) {
        self.refresh_providers(false, cx);
    }

    fn discover(&mut self, cx: &mut Context<Self>) {
        self.refresh_providers(true, cx);
    }

    fn refresh_providers(&mut self, discover: bool, cx: &mut Context<Self>) {
        #[cfg(test)]
        let started = std::time::Instant::now();
        let generation = self.host.start_refresh();
        self.cancel_projection();
        self.refresh_pending = true;
        let store = self.project.read(cx).structured_execution_store().clone();
        let mut discoveries = if discover {
            self.delegate.discover(&self.project, cx)
        } else {
            Vec::new()
        }
        .into_iter()
        .collect::<HashMap<_, _>>();
        let mut provider_ids = self.provider_ids.clone();
        provider_ids.extend(discoveries.keys().cloned());
        provider_ids.extend(
            store
                .read(cx)
                .state()
                .providers()
                .map(|provider| provider.provider_id.clone()),
        );
        provider_ids.sort();
        provider_ids.dedup();
        self.provider_ids = provider_ids.clone();
        if provider_ids.is_empty() {
            self.snapshots = Arc::default();
            self.refresh_pending = false;
            self.refresh_status = None;
            self.projection = TestExplorerProjection::default();
            self.host
                .apply_refresh(generation, Ok(self.projection.snapshot.clone()));
            cx.notify();
            return;
        }
        let refreshes = provider_ids
            .into_iter()
            .map(|provider_id| {
                discoveries.remove(&provider_id).unwrap_or_else(|| {
                    if discover {
                        store.update(cx, |store, cx| store.refresh_provider(provider_id, cx))
                    } else {
                        gpui::Task::ready(
                            store
                                .read(cx)
                                .state()
                                .provider(&provider_id)
                                .cloned()
                                .ok_or_else(|| {
                                    anyhow::anyhow!("No cached structured test provider")
                                }),
                        )
                    }
                })
            })
            .collect::<Vec<_>>();
        let task = cx.spawn(async move |this, cx| {
            let results = futures::future::join_all(refreshes).await;
            this.update(cx, |panel, cx| {
                if panel.host.refresh_generation() != generation {
                    return;
                }
                panel.refresh_pending = false;
                let mut snapshots = Vec::new();
                let mut first_error = None;
                for result in results {
                    match result {
                        Ok(snapshot) => snapshots.push(snapshot),
                        Err(error) if first_error.is_none() => first_error = Some(error),
                        Err(_) => {}
                    }
                }
                if snapshots.is_empty() {
                    panel.cancel_projection();
                    let error = first_error.unwrap_or_else(|| {
                        anyhow::anyhow!("No structured test provider returned a snapshot")
                    });
                    let status = LanguageToolProviderStatus::Error(error.to_string());
                    if panel.host.apply_refresh(generation, Err(error)) {
                        panel.refresh_status = Some(status);
                    }
                } else {
                    panel.apply_provider_snapshots(generation, snapshots.into(), first_error, cx);
                }
                cx.notify();
            })
            .ok();
        });
        self.host.replace_refresh_task(task);
        cx.notify();
        #[cfg(test)]
        {
            self.projection_timings.refresh += started.elapsed();
        }
    }

    fn cancel_projection(&mut self) {
        self.projection_generation = self.projection_generation.wrapping_add(1);
        self.projection_task = gpui::Task::ready(());
    }

    fn rebuild_projection(&mut self, cx: &mut Context<Self>) {
        self.cancel_projection();
        let generation = self.projection_generation;
        let refresh_generation = self.host.refresh_generation();
        let snapshots = self.snapshots.clone();
        let filter = self.filter.clone();
        let refresh_status = self.refresh_status.clone();
        let refresh_pending = self.refresh_pending;
        #[cfg(test)]
        let background_executor = cx.background_executor().clone();
        #[cfg(test)]
        let background_nanoseconds = self.projection_timings.background_nanoseconds.clone();
        #[cfg(test)]
        let background_gate = self.projection_timings.background_gate.take();
        let projected = cx.background_spawn(async move {
            #[cfg(test)]
            if let Some(gate) = background_gate {
                gate.await.expect("projection gate released");
            }
            #[cfg(test)]
            let started = std::time::Instant::now();
            #[cfg(test)]
            assert!(
                !background_executor.is_main_thread(),
                "Tests projection must run in the background"
            );
            let mut projection = project_test_explorer(&snapshots, &filter);
            if refresh_pending {
                projection.snapshot.status = LanguageToolProviderStatus::Loading;
            } else if let Some(status) = refresh_status {
                projection.snapshot.status = status;
            }
            let tree = projection.snapshot.clone();
            #[cfg(test)]
            background_nanoseconds.store(
                started
                    .elapsed()
                    .as_nanos()
                    .try_into()
                    .expect("bounded projection duration"),
                std::sync::atomic::Ordering::SeqCst,
            );
            (projection, tree)
        });
        self.projection_task = cx.spawn(async move |this, cx| {
            let (projection, tree) = projected.await;
            this.update(cx, |panel, cx| {
                panel.apply_projection(generation, refresh_generation, projection, tree, cx);
            })
            .log_err();
        });
        cx.notify();
    }

    fn apply_projection(
        &mut self,
        generation: u64,
        refresh_generation: u64,
        projection: TestExplorerProjection,
        tree: LanguageToolSnapshot,
        cx: &mut Context<Self>,
    ) {
        if self.projection_generation != generation
            || self.host.refresh_generation() != refresh_generation
        {
            return;
        }
        #[cfg(test)]
        let started = std::time::Instant::now();
        self.host.apply_refresh(refresh_generation, Ok(tree));
        self.projection = projection;
        self.restore_selection();
        cx.notify();
        #[cfg(test)]
        {
            self.projection_timings.application += started.elapsed();
        }
    }

    fn apply_provider_snapshots(
        &mut self,
        generation: u64,
        snapshots: Arc<[StructuredProviderSnapshot]>,
        error: Option<anyhow::Error>,
        cx: &mut Context<Self>,
    ) {
        if self.host.refresh_generation() != generation {
            return;
        }
        #[cfg(test)]
        let started = std::time::Instant::now();
        self.refresh_pending = false;
        self.snapshots = snapshots;
        self.host.take_dirty();
        self.refresh_status = error.map(|error| {
            LanguageToolProviderStatus::Partial(format!(
                "Some test providers could not refresh: {error}"
            ))
        });
        self.rebuild_projection(cx);
        #[cfg(test)]
        {
            self.projection_timings.admission += started.elapsed();
        }
    }

    fn restore_selection(&mut self) {
        let Some((provider_id, node_id)) = self.restored_selection.take() else {
            return;
        };
        self.host.select(tree_id(&provider_id, &node_id));
    }

    fn selected(&self) -> Option<&TestExplorerSelection> {
        self.host
            .selected()
            .and_then(|id| self.projection.selection(id))
    }

    fn set_status_filter(&mut self, status: TestStatusFilter, cx: &mut Context<Self>) {
        self.filter.status = status;
        self.rebuild_projection(cx);
        self.persist(cx);
    }

    fn current_source_path(&self, id: &LanguageToolNodeId, cx: &App) -> Option<ProjectPath> {
        let selection = self.projection.selection(id)?;
        let projected_path = self.projection.navigation(id)?;
        self.project.read_with(cx, |project, cx| {
            if project.is_disconnected(cx)
                || project::trusted_worktrees::TrustedWorktrees::has_restricted_worktrees(
                    &project.worktree_store(),
                    cx,
                )
            {
                return None;
            }
            let store = project.structured_execution_store().read(cx);
            let provider = store.state().provider(&selection.provider_id)?;
            if provider.discovery_generation != selection.discovery_generation
                || !matches!(
                    provider.status,
                    StructuredProviderStatus::Current | StructuredProviderStatus::Partial
                )
            {
                return None;
            }
            let node = provider
                .nodes
                .iter()
                .find(|node| node.id == selection.node_id)?;
            let run = provider
                .current_run
                .as_ref()
                .or(provider.last_complete_run.as_ref());
            let path = run
                .and_then(|run| {
                    run.events
                        .iter()
                        .rev()
                        .find(|event| event.node_id == selection.node_id)
                })
                .and_then(|event| event.location.clone())
                .or_else(|| node.path.clone())?;
            if &path != projected_path
                || !project
                    .visible_worktrees(cx)
                    .any(|worktree| worktree.read(cx).id() == path.worktree_id)
                || !project.entry_for_path(&path, cx).is_some_and(|entry| {
                    entry.is_file() && !entry.is_private && !entry.is_external && !entry.is_fifo
                })
            {
                return None;
            }
            Some(path)
        })
    }

    fn activate_selected(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        if !self.host.can_activate() {
            return;
        }
        let Some(path) = self
            .host
            .selected()
            .and_then(|id| self.current_source_path(id, cx))
        else {
            return;
        };
        if let Some(workspace) = self.workspace.upgrade() {
            workspace
                .update(cx, |workspace, cx| {
                    workspace.open_path(path, None, true, window, cx)
                })
                .detach_and_log_err(cx);
        }
    }

    fn run_selected(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        if let Some(reason) = self
            .host
            .selected()
            .and_then(|id| self.projection.experimental_disabled_reason(id, cx))
        {
            self.set_notice(reason, cx);
            return;
        }
        if !self
            .host
            .selected()
            .is_some_and(|id| self.projection.can_run_with_flags(id, cx))
        {
            self.set_notice("Select a test, group, or suite to run", cx);
            return;
        }
        let Some(selection) = self.selected().cloned() else {
            self.set_notice("Select a test, group, or suite to run", cx);
            return;
        };
        let delegate = self.delegate.clone();
        let workspace = self.workspace.clone();
        self.handle_action_result(delegate.run(&selection, &workspace, window, cx), cx);
    }

    fn debug_selected(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        let disabled_reason = self
            .host
            .selected()
            .map(|id| self.projection.debug_disabled_reason_with_flags(id, cx))
            .unwrap_or(Some("Select an individual test case to debug".to_string()));
        if let Some(reason) = disabled_reason {
            self.set_notice(reason, cx);
            return;
        }
        let Some(selection) = self.selected().cloned() else {
            self.set_notice("Select an individual test case to debug", cx);
            return;
        };
        let delegate = self.delegate.clone();
        let workspace = self.workspace.clone();
        self.handle_action_result(delegate.debug(&selection, &workspace, window, cx), cx);
    }

    fn cancel_selected(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        let Some(selection) = self.selected().cloned().or_else(|| {
            self.projection
                .selections
                .values()
                .find(|selection| selection.run_id.is_some())
                .cloned()
        }) else {
            self.set_notice("No structured test run is active", cx);
            return;
        };
        let delegate = self.delegate.clone();
        let workspace = self.workspace.clone();
        self.handle_action_result(delegate.cancel(&selection, &workspace, window, cx), cx);
    }

    fn rerun_failed(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        if self.projection.failed.is_empty() {
            self.set_notice("No failed tests are available to rerun", cx);
            return;
        }
        let failed = self.projection.enabled_failed(cx);
        if failed.is_empty() {
            self.set_notice("No enabled failed test actions are available to rerun", cx);
            return;
        }
        let delegate = self.delegate.clone();
        let workspace = self.workspace.clone();
        self.handle_action_result(delegate.rerun_failed(&failed, &workspace, window, cx), cx);
    }

    fn reveal_terminal(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        let Some(selection) = self.selected().cloned() else {
            self.set_notice("Select a test result with an owning task terminal", cx);
            return;
        };
        let delegate = self.delegate.clone();
        let workspace = self.workspace.clone();
        self.handle_action_result(
            delegate.reveal_terminal(&selection, &workspace, window, cx),
            cx,
        );
    }

    fn handle_action_result(&mut self, result: anyhow::Result<()>, cx: &mut Context<Self>) {
        match result {
            Ok(()) => self.action_notice = None,
            Err(error) => self.action_notice = Some(error.to_string()),
        }
        cx.notify();
    }

    fn set_notice(&mut self, notice: impl Into<String>, cx: &mut Context<Self>) {
        self.action_notice = Some(notice.into());
        cx.notify();
    }

    fn persist(&mut self, cx: &mut Context<Self>) {
        let Some(key) = self.state_key.clone() else {
            return;
        };
        let selected = self.selected().cloned();
        self.serialization_task = persist_state(
            key,
            PersistedTestsPanelState {
                version: TESTS_PANEL_STATE_VERSION,
                query: self.filter.query.clone(),
                status: self.filter.status,
                selected_provider: selected
                    .as_ref()
                    .map(|selection| selection.provider_id.0.clone()),
                selected_node: selected.map(|selection| selection.node_id.0),
            },
            cx,
        );
    }

    fn deploy_context_menu(
        &mut self,
        position: Point<Pixels>,
        id: LanguageToolNodeId,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        self.host.select(id);
        let can_open = self
            .host
            .selected()
            .is_some_and(|id| self.current_source_path(id, cx).is_some());
        let has_selection = self.selected().is_some();
        let can_run = self
            .host
            .selected()
            .is_some_and(|id| self.projection.can_run_with_flags(id, cx));
        let can_debug = self
            .host
            .selected()
            .is_some_and(|id| self.projection.can_debug_with_flags(id, cx));
        let run_label = self
            .host
            .selected()
            .and_then(|id| self.projection.experimental_disabled_reason(id, cx))
            .map(|reason| format!("Run: {reason}"))
            .unwrap_or_else(|| "Run".to_string());
        let debug_label = self
            .host
            .selected()
            .and_then(|id| self.projection.debug_disabled_reason_with_flags(id, cx))
            .map(|reason| format!("Debug: {reason}"))
            .unwrap_or_else(|| "Debug".to_string());
        let has_failed = !self.projection.enabled_failed(cx).is_empty();
        let menu = ContextMenu::build(window, cx, |menu, _, _| {
            menu.context(self.focus_handle.clone())
                .action_disabled_when(!can_run, run_label, Box::new(RunSelectedTests))
                .action_disabled_when(!can_debug, debug_label, Box::new(DebugSelectedTests))
                .action_disabled_when(!has_selection, "Cancel", Box::new(CancelTestRun))
                .action_disabled_when(!has_failed, "Rerun Failed", Box::new(RerunFailedTests))
                .action_disabled_when(
                    !has_selection,
                    "Show Task Terminal",
                    Box::new(RevealTestTerminal),
                )
                .separator()
                .when(can_open, |menu| {
                    menu.action(
                        "Open Location",
                        Box::new(language_tool_tree::ActivateSelected),
                    )
                })
                .action("Refresh", Box::new(language_tool_tree::Refresh))
                .separator()
                .action("Expand All", Box::new(language_tool_tree::ExpandAll))
                .action("Collapse All", Box::new(language_tool_tree::CollapseAll))
        });
        window.focus(&menu.focus_handle(cx), cx);
        let subscription = cx.subscribe(&menu, |panel, _, _: &DismissEvent, cx| {
            panel.context_menu.take();
            cx.notify();
        });
        self.context_menu = Some((menu, position, subscription));
        self.persist(cx);
        cx.notify();
    }
}

impl Panel for TestsPanel {
    fn persistent_name() -> &'static str {
        "Tests"
    }

    fn panel_key() -> &'static str {
        TESTS_PANEL_KEY
    }

    fn position(&self, _: &Window, cx: &App) -> DockPosition {
        match TestsPanelSettings::get_global(cx).dock {
            DockSide::Left => DockPosition::Left,
            DockSide::Right => DockPosition::Right,
        }
    }

    fn position_is_valid(&self, position: DockPosition) -> bool {
        matches!(position, DockPosition::Left | DockPosition::Right)
    }

    fn set_position(&mut self, position: DockPosition, _: &mut Window, cx: &mut Context<Self>) {
        settings::update_settings_file(<dyn fs::Fs>::global(cx), cx, move |settings, _| {
            settings.tests_panel.get_or_insert_default().dock = Some(match position {
                DockPosition::Right => DockSide::Right,
                DockPosition::Left | DockPosition::Bottom => DockSide::Left,
            });
        });
    }

    fn default_size(&self, _: &Window, cx: &App) -> Pixels {
        TestsPanelSettings::get_global(cx).default_width
    }

    fn icon(&self, _: &Window, cx: &App) -> Option<IconName> {
        TestsPanelSettings::get_global(cx)
            .button
            .then_some(IconName::ListTodo)
    }

    fn icon_tooltip(&self, _: &Window, _: &App) -> Option<&'static str> {
        Some("Tests")
    }

    fn toggle_action(&self) -> Box<dyn Action> {
        Box::new(ToggleTestsPanel)
    }

    fn starts_open(&self, _: &Window, cx: &App) -> bool {
        TestsPanelSettings::get_global(cx).starts_open
    }

    fn set_active(&mut self, active: bool, _: &mut Window, cx: &mut Context<Self>) {
        let was_active = self.active;
        self.active = active;
        if active
            && !was_active
            && (matches!(self.host.status(), LanguageToolTreeStatus::Dormant)
                || self.host.take_dirty())
        {
            #[cfg(feature = "jvm-test-actions")]
            if self.java_discovery_pending {
                let store = self.project.read(cx).structured_execution_store().clone();
                let has_active_run = store.read(cx).state().providers().any(|snapshot| {
                    snapshot
                        .current_run
                        .as_ref()
                        .is_some_and(|run| !run.phase.is_terminal())
                });
                if has_active_run {
                    self.host.mark_dirty();
                    return;
                }
                self.java_discovery_pending = false;
            }
            self.discover(cx);
        }
    }

    fn activation_priority(&self) -> u32 {
        8
    }

    fn hide_button_setting(&self, _: &App) -> Option<workspace::HideStatusItem> {
        Some(workspace::HideStatusItem::new(|settings| {
            settings.tests_panel.get_or_insert_default().button = Some(false);
        }))
    }
}

impl Focusable for TestsPanel {
    fn focus_handle(&self, _: &App) -> FocusHandle {
        self.focus_handle.clone()
    }
}

impl EventEmitter<PanelEvent> for TestsPanel {}

impl Render for TestsPanel {
    fn render(&mut self, _window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let rows = self.host.visible_rows_snapshot();
        let selected = self.host.selected().cloned();
        let status = status_message(self.host.status());
        let can_expand_all = self.host.can_expand_all();
        let can_collapse_all = self.host.can_collapse_all();
        let can_refresh = self.host.can_refresh();
        let has_selection = self.selected().is_some();
        let can_run = selected
            .as_ref()
            .is_some_and(|id| self.projection.can_run_with_flags(id, cx));
        let can_debug = selected
            .as_ref()
            .is_some_and(|id| self.projection.can_debug_with_flags(id, cx));
        let debug_disabled_reason = selected
            .as_ref()
            .and_then(|id| self.projection.debug_disabled_reason_with_flags(id, cx))
            .or_else(|| {
                (!has_selection).then(|| "Select an individual test case to debug".to_string())
            });
        let debug_tooltip = debug_disabled_reason
            .clone()
            .unwrap_or_else(|| "Debug Selected Test".to_string());
        let run_disabled_reason = selected
            .as_ref()
            .and_then(|id| self.projection.experimental_disabled_reason(id, cx));
        let run_tooltip = run_disabled_reason
            .clone()
            .unwrap_or_else(|| "Run Selected Tests".to_string());
        let has_failed = !self.projection.enabled_failed(cx).is_empty();
        let has_run = self
            .projection
            .selections
            .values()
            .any(|selection| selection.run_id.is_some());
        let summary_label = self.projection.summary_label();
        let click_panel = cx.weak_entity();
        let toggle_panel = cx.weak_entity();
        let context_panel = cx.weak_entity();
        v_flex()
            .id("tests-panel")
            .role(Role::Group)
            .aria_label("Tests panel")
            .key_context("TestsPanel")
            .track_focus(&self.focus_handle)
            .size_full()
            .on_action(
                cx.listener(|panel, _: &language_tool_tree::Refresh, _, cx| panel.discover(cx)),
            )
            .on_action(
                cx.listener(|panel, _: &language_tool_tree::ExpandAll, _, cx| {
                    panel.host.expand_all();
                    cx.notify();
                }),
            )
            .on_action(
                cx.listener(|panel, _: &language_tool_tree::CollapseAll, _, cx| {
                    panel.host.collapse_all();
                    cx.notify();
                }),
            )
            .on_action(
                cx.listener(|panel, _: &language_tool_tree::SelectNext, _, cx| {
                    panel.host.select_next();
                    panel.host.reveal_selection(ScrollStrategy::Center);
                    panel.persist(cx);
                    cx.notify();
                }),
            )
            .on_action(
                cx.listener(|panel, _: &language_tool_tree::SelectPrevious, _, cx| {
                    panel.host.select_previous();
                    panel.host.reveal_selection(ScrollStrategy::Center);
                    panel.persist(cx);
                    cx.notify();
                }),
            )
            .on_action(
                cx.listener(|panel, _: &language_tool_tree::SelectFirst, _, cx| {
                    panel.host.select_first();
                    panel.host.reveal_selection(ScrollStrategy::Top);
                    panel.persist(cx);
                    cx.notify();
                }),
            )
            .on_action(
                cx.listener(|panel, _: &language_tool_tree::SelectLast, _, cx| {
                    panel.host.select_last();
                    panel.host.reveal_selection(ScrollStrategy::Bottom);
                    panel.persist(cx);
                    cx.notify();
                }),
            )
            .on_action(
                cx.listener(|panel, _: &language_tool_tree::SelectParent, _, cx| {
                    panel.host.select_parent();
                    panel.host.reveal_selection(ScrollStrategy::Center);
                    panel.persist(cx);
                    cx.notify();
                }),
            )
            .on_action(
                cx.listener(|panel, _: &language_tool_tree::SelectFirstChild, _, cx| {
                    panel.host.select_first_child();
                    panel.host.reveal_selection(ScrollStrategy::Center);
                    panel.persist(cx);
                    cx.notify();
                }),
            )
            .on_action(
                cx.listener(|panel, _: &language_tool_tree::ToggleExpanded, _, cx| {
                    panel.host.toggle_selected();
                    cx.notify();
                }),
            )
            .on_action(cx.listener(
                |panel, _: &language_tool_tree::ActivateSelected, window, cx| {
                    panel.activate_selected(window, cx)
                },
            ))
            .on_action(
                cx.listener(|panel, _: &RunSelectedTests, window, cx| {
                    panel.run_selected(window, cx)
                }),
            )
            .on_action(cx.listener(|panel, _: &DebugSelectedTests, window, cx| {
                panel.debug_selected(window, cx)
            }))
            .on_action(
                cx.listener(|panel, _: &CancelTestRun, window, cx| {
                    panel.cancel_selected(window, cx)
                }),
            )
            .on_action(
                cx.listener(|panel, _: &RerunFailedTests, window, cx| {
                    panel.rerun_failed(window, cx)
                }),
            )
            .on_action(cx.listener(|panel, _: &RevealTestTerminal, window, cx| {
                panel.reveal_terminal(window, cx)
            }))
            .child(
                div()
                    .min_h_9()
                    .px_2()
                    .py_1()
                    .flex()
                    .flex_wrap()
                    .gap_1()
                    .items_center()
                    .justify_between()
                    .child("Tests")
                    .child(
                        h_flex()
                            .min_w_0()
                            .flex_wrap()
                            .gap_1()
                            .child(
                                IconButton::new("tests-run", IconName::PlayFilled)
                                    .shape(ui::IconButtonShape::Square)
                                    .aria_label("Run Selected Tests")
                                    .tooltip(Tooltip::text(run_tooltip))
                                    .when_some(run_disabled_reason, |button, reason| {
                                        button.aria_description(reason)
                                    })
                                    .disabled(!can_run)
                                    .on_click(cx.listener(|panel, _, window, cx| {
                                        panel.run_selected(window, cx)
                                    })),
                            )
                            .child(
                                IconButton::new("tests-debug", IconName::Debug)
                                    .shape(ui::IconButtonShape::Square)
                                    .aria_label("Debug Selected Test")
                                    .tooltip(Tooltip::text(debug_tooltip))
                                    .when_some(debug_disabled_reason, |button, reason| {
                                        button.aria_description(reason)
                                    })
                                    .disabled(!can_debug)
                                    .on_click(cx.listener(|panel, _, window, cx| {
                                        panel.debug_selected(window, cx)
                                    })),
                            )
                            .child(
                                IconButton::new("tests-cancel", IconName::Stop)
                                    .shape(ui::IconButtonShape::Square)
                                    .aria_label("Cancel Test Run")
                                    .tooltip(Tooltip::text("Cancel Test Run"))
                                    .disabled(!has_run)
                                    .on_click(cx.listener(|panel, _, window, cx| {
                                        panel.cancel_selected(window, cx)
                                    })),
                            )
                            .child(
                                IconButton::new("tests-rerun-failed", IconName::Rerun)
                                    .shape(ui::IconButtonShape::Square)
                                    .aria_label("Rerun Failed Tests")
                                    .tooltip(Tooltip::text("Rerun Failed Tests"))
                                    .disabled(!has_failed)
                                    .on_click(cx.listener(|panel, _, window, cx| {
                                        panel.rerun_failed(window, cx)
                                    })),
                            )
                            .child(
                                IconButton::new("tests-terminal", IconName::TerminalAlt)
                                    .shape(ui::IconButtonShape::Square)
                                    .aria_label("Show Task Terminal")
                                    .tooltip(Tooltip::text("Show Task Terminal"))
                                    .disabled(!has_selection)
                                    .on_click(cx.listener(|panel, _, window, cx| {
                                        panel.reveal_terminal(window, cx)
                                    })),
                            )
                            .child(
                                IconButton::new("tests-expand-all", IconName::ExpandVertical)
                                    .shape(ui::IconButtonShape::Square)
                                    .aria_label("Expand All")
                                    .tooltip(Tooltip::text("Expand All"))
                                    .disabled(!can_expand_all)
                                    .on_click(cx.listener(|panel, _, _, cx| {
                                        panel.host.expand_all();
                                        cx.notify();
                                    })),
                            )
                            .child(
                                IconButton::new("tests-collapse-all", IconName::ListCollapse)
                                    .shape(ui::IconButtonShape::Square)
                                    .aria_label("Collapse All")
                                    .tooltip(Tooltip::text("Collapse All"))
                                    .disabled(!can_collapse_all)
                                    .on_click(cx.listener(|panel, _, _, cx| {
                                        panel.host.collapse_all();
                                        cx.notify();
                                    })),
                            )
                            .child(
                                IconButton::new("tests-refresh", IconName::RefreshTitle)
                                    .shape(ui::IconButtonShape::Square)
                                    .aria_label("Refresh Tests")
                                    .tooltip(Tooltip::text("Refresh Tests"))
                                    .disabled(!can_refresh)
                                    .on_click(cx.listener(|panel, _, _, cx| panel.discover(cx))),
                            ),
                    ),
            )
            .child(div().px_2().pb_1().child(self.query_input.clone()))
            .child(
                h_flex().flex_wrap().px_2().pb_1().gap_1().children(
                    [
                        TestStatusFilter::All,
                        TestStatusFilter::Failed,
                        TestStatusFilter::Passed,
                        TestStatusFilter::Skipped,
                        TestStatusFilter::Running,
                        TestStatusFilter::Cancelled,
                    ]
                    .into_iter()
                    .map(|status| {
                        Button::new(
                            format!("tests-filter-{}", status.label().to_lowercase()),
                            status.label(),
                        )
                        .toggle_state(self.filter.status == status)
                        .on_click(
                            cx.listener(move |panel, _, _, cx| panel.set_status_filter(status, cx)),
                        )
                    }),
                ),
            )
            .child(
                div()
                    .id("tests-summary")
                    .px_2()
                    .py_1()
                    .text_color(cx.theme().colors().text_muted)
                    .role(Role::Status)
                    .aria_label("Test results")
                    .aria_value(summary_label.clone())
                    .aria_live(gpui::accesskit::Live::Polite)
                    .child(summary_label),
            )
            .when_some(self.action_notice.clone(), |panel, notice| {
                panel.child(
                    div()
                        .id("tests-action-status")
                        .role(Role::Status)
                        .aria_label("Test action status")
                        .aria_value(notice.clone())
                        .aria_live(gpui::accesskit::Live::Polite)
                        .px_2()
                        .py_1()
                        .text_color(cx.theme().status().warning)
                        .child(notice),
                )
            })
            .child(language_tool_tree(
                rows,
                selected,
                status,
                self.host.scroll_handle().clone(),
                move |id, click_count, window, cx| {
                    if let Err(error) = click_panel.update(cx, |panel, cx| {
                        window.focus(&panel.focus_handle, cx);
                        panel.host.select(id.clone());
                        panel.persist(cx);
                        if click_count > 1 {
                            panel.activate_selected(window, cx);
                        }
                        cx.notify();
                    }) {
                        log::debug!("Tests panel was dropped: {error:#}");
                    }
                },
                move |id, _, cx| {
                    if let Err(error) = toggle_panel.update(cx, |panel, cx| {
                        panel.host.select(id.clone());
                        panel.host.toggle(&id);
                        cx.notify();
                    }) {
                        log::debug!("Tests panel was dropped: {error:#}");
                    }
                },
                move |id, position, window, cx| {
                    if let Err(error) = context_panel.update(cx, |panel, cx| {
                        panel.deploy_context_menu(position, id, window, cx)
                    }) {
                        log::debug!("Tests panel was dropped: {error:#}");
                    }
                },
            ))
            .children(self.context_menu.as_ref().map(|(menu, position, _)| {
                deferred(
                    anchored()
                        .position(*position)
                        .anchor(Anchor::TopLeft)
                        .child(menu.clone()),
                )
                .with_priority(1)
            }))
    }
}

pub fn init(cx: &mut App) {
    SettingsStore::update_global(cx, |store, cx| {
        store.update_default_settings(cx, |settings| {
            settings.tests_panel = Some(TestsPanelSettingsContent {
                button: Some(true),
                default_width: Some(300.),
                dock: Some(DockSide::Left),
                starts_open: Some(false),
            });
        });
        store.register_setting::<TestsPanelSettings>();
    });
    cx.observe_new(|workspace: &mut Workspace, _, _| {
        workspace.register_action(TestsPanel::toggle_focus);
    })
    .detach();
    bind_test_panel_keys(cx);
}

pub fn bind_test_panel_keys(cx: &mut App) {
    cx.bind_keys([
        gpui::KeyBinding::new("up", language_tool_tree::SelectPrevious, Some("TestsPanel")),
        gpui::KeyBinding::new("down", language_tool_tree::SelectNext, Some("TestsPanel")),
        gpui::KeyBinding::new("home", language_tool_tree::SelectFirst, Some("TestsPanel")),
        gpui::KeyBinding::new("end", language_tool_tree::SelectLast, Some("TestsPanel")),
        gpui::KeyBinding::new("left", language_tool_tree::SelectParent, Some("TestsPanel")),
        gpui::KeyBinding::new(
            "right",
            language_tool_tree::SelectFirstChild,
            Some("TestsPanel"),
        ),
        gpui::KeyBinding::new(
            "space",
            language_tool_tree::ToggleExpanded,
            Some("TestsPanel"),
        ),
        gpui::KeyBinding::new(
            "enter",
            language_tool_tree::ActivateSelected,
            Some("TestsPanel"),
        ),
        gpui::KeyBinding::new(
            "cmd-shift-e",
            language_tool_tree::ExpandAll,
            Some("TestsPanel"),
        ),
        gpui::KeyBinding::new(
            "cmd-shift-c",
            language_tool_tree::CollapseAll,
            Some("TestsPanel"),
        ),
        gpui::KeyBinding::new("cmd-r", language_tool_tree::Refresh, Some("TestsPanel")),
    ]);
}

#[cfg(all(
    feature = "jvm-test-actions",
    any(test, feature = "jvm-workflow-test-support")
))]
pub mod jvm_workflow;

#[cfg(test)]
mod tests {
    use std::{
        collections::VecDeque,
        time::{Duration, Instant},
    };

    use gpui::TestAppContext;
    use settings::WorktreeId;
    use util::rel_path::RelPath;

    use super::*;

    #[cfg(all(unix, feature = "rust-test-actions"))]
    #[gpui::test]
    async fn rust_test_task_arguments_reach_shell_literally(cx: &mut TestAppContext) {
        cx.executor().allow_parking();
        let result: anyhow::Result<()> = async {
            let arguments = [
                "src/lib.rs - add (line 3)",
                "module::\"quoted\"",
                "module::'quoted'",
                "$RUST_TEST_LITERAL_VALUE",
                "$ZED_WORKTREE_ROOT",
                "${ZED_WORKTREE_ROOT}",
                "$ZED_UNKNOWN_TEST_VALUE",
                "$$",
                "$(printf injected)",
                "value; printf injected",
                "path\\name",
                "",
                "unicode::例",
            ];
            let template = task::TaskTemplate {
                label: "Rust test $ZED_UNKNOWN_TEST_VALUE".into(),
                command: "/usr/bin/printf".into(),
                args: std::iter::once("%s\n".to_string())
                    .chain(arguments.iter().map(|argument| argument.to_string()))
                    .collect(),
                cwd: Some(task::VariableName::WorktreeRoot.template_value()),
                ..Default::default()
            };
            let source = TaskSourceKind::Language {
                name: "Rust Tests".into(),
            };
            let mut context = task::TaskContext::default();
            context
                .task_variables
                .insert(task::VariableName::WorktreeRoot, "/tmp/rust-fixture".into());
            let resolved = resolve_rust_test_task(&template, &source, &context)?;
            assert_eq!(resolved.resolved_label, template.label);
            assert_eq!(
                resolved.resolved.cwd.as_deref(),
                Some(std::path::Path::new("/tmp/rust-fixture"))
            );
            let expected = format!("{}\n", arguments.join("\n"));
            let shells = [
                "/bin/sh",
                "/bin/bash",
                #[cfg(target_os = "macos")]
                "/bin/zsh",
            ];
            for shell in shells {
                let prepared = terminal_view::terminal_panel::prepare_task_for_spawn(
                    &resolved.resolved,
                    &task::Shell::Program(shell.into()),
                    false,
                );
                let mut command = std::process::Command::new(
                    prepared.command.context("prepared Rust test shell")?,
                );
                command
                    .args(prepared.args)
                    .env("RUST_TEST_LITERAL_VALUE", "expanded");
                let output = util::process::Child::spawn(
                    command,
                    std::process::Stdio::null(),
                    std::process::Stdio::piped(),
                    std::process::Stdio::piped(),
                )?
                .output()
                .await?;
                assert!(
                    output.status.success(),
                    "{shell}: {}",
                    String::from_utf8_lossy(&output.stderr)
                );
                assert_eq!(String::from_utf8(output.stdout)?, expected, "{shell}");

                let user_template = task::TaskTemplate {
                    label: "Ordinary user Task".into(),
                    args: vec!["%s".into(), "$RUST_TEST_LITERAL_VALUE".into()],
                    ..template.clone()
                };
                let user_task = user_template
                    .resolve_task(&source.to_id_base(), &context)
                    .context("ordinary user Task")?;
                let prepared = terminal_view::terminal_panel::prepare_task_for_spawn(
                    &user_task.resolved,
                    &task::Shell::Program(shell.into()),
                    false,
                );
                let mut command = std::process::Command::new(
                    prepared.command.context("prepared ordinary Task shell")?,
                );
                command
                    .args(prepared.args)
                    .env("RUST_TEST_LITERAL_VALUE", "expanded");
                let output = util::process::Child::spawn(
                    command,
                    std::process::Stdio::null(),
                    std::process::Stdio::piped(),
                    std::process::Stdio::piped(),
                )?
                .output()
                .await?;
                assert!(output.status.success());
                assert_eq!(output.stdout, b"expanded", "{shell}");
            }
            Ok(())
        }
        .await;
        assert!(result.is_ok(), "{result:?}");
    }

    #[cfg(feature = "jvm-test-actions")]
    use language::{LanguageServerId, LanguageServerName};
    #[cfg(feature = "jvm-test-actions")]
    use project::jvm_test_provider::JvmTestBuildTool;

    #[cfg(all(feature = "jvm-test-actions", target_os = "macos"))]
    use super::jvm_workflow::selected_java_case_through_tests_panel_with_connection;
    #[cfg(feature = "jvm-test-actions")]
    use super::jvm_workflow::{JavaTestOutcome, selected_java_case_through_tests_panel};

    #[cfg(feature = "jvm-test-actions")]
    #[gpui::test]
    async fn default_delegate_discovers_java_tests_without_hiding_rust(cx: &mut TestAppContext) {
        crate::tests::init_test(cx);
        let fs = fs::FakeFs::new(cx.executor());
        fs.insert_tree(
            std::path::Path::new("/fixture"),
            serde_json::json!({
                "pom.xml": "<project/>",
                "src": { "test": { "java": { "GreetingTest.java":
                    "class GreetingTest { @Test messageIsAvailable() {} }"
                } } },
                "target": { "surefire-reports": { "TEST-GreetingTest.xml":
                    "<testsuite name=\"GreetingTest\"><testcase classname=\"GreetingTest\" name=\"messageIsAvailable\"/></testsuite>"
                } }
            }),
        )
        .await;
        let project = Project::test(fs, [std::path::Path::new("/fixture")], cx).await;
        let delegate = default_test_explorer_delegate();
        let discoveries = cx.update(|cx| delegate.discover(&project, cx));
        let expected_count = if cfg!(feature = "rust-test-actions") {
            2
        } else {
            1
        };
        assert_eq!(discoveries.len(), expected_count);
        let java = discoveries
            .into_iter()
            .find(|(provider_id, _)| provider_id.0 == JVM_JUNIT_PROVIDER_ID)
            .expect("Java provider discovery");
        let snapshot = java.1.await.expect("Java discovery snapshot");
        assert_eq!(snapshot.provider_id.0, JVM_JUNIT_PROVIDER_ID);
        let projection = project_test_explorer(&[snapshot], &TestExplorerFilter::default());
        assert!(find_node(&projection.snapshot.roots, "Java JUnit tests").is_some());
        assert!(find_node(&projection.snapshot.roots, "messageIsAvailable").is_some());
    }

    #[cfg(feature = "jvm-test-actions")]
    #[gpui::test]
    async fn selected_java_case_reaches_the_trust_gated_task_path(cx: &mut TestAppContext) {
        use std::borrow::Cow;

        use language::{
            Language, LanguageConfig, LanguageMatcher, LanguageQueries, tree_sitter_typescript,
        };

        crate::tests::init_test(cx);
        let fs = fs::FakeFs::new(cx.executor());
        fs.insert_tree(
            std::path::Path::new("/fixture"),
            serde_json::json!({
                "pom.xml": "<project/>",
                "mvnw": "#!/bin/sh\n",
                "src": { "test": { "java": { "GreetingTest.java":
                    "class GreetingTest { @Test messageIsAvailable() {} }"
                } } },
                "target": { "surefire-reports": { "TEST-GreetingTest.xml":
                    "<testsuite name=\"GreetingTest\"><testcase classname=\"GreetingTest\" name=\"messageIsAvailable\"/></testsuite>"
                } }
            }),
        )
        .await;
        let project = Project::test(fs, [std::path::Path::new("/fixture")], cx).await;
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
                "(program (class_declaration name: (type_identifier) @java_class_name body: (class_body (decorator (identifier) @annotation_name) (method_definition name: (property_identifier) @run @java_method_name))) @_ (#set! tag java-test-method))",
            )),
            ..Default::default()
        })
        .expect("fixture runnable query");
        languages.add(Arc::new(java));
        let delegate = default_test_explorer_delegate();
        let discoveries = cx.update(|cx| delegate.discover(&project, cx));
        let snapshot = discoveries
            .into_iter()
            .find(|(provider_id, _)| provider_id.0 == JVM_JUNIT_PROVIDER_ID)
            .expect("Java discovery")
            .1
            .await
            .expect("source-backed Java snapshot");
        let projection = project_test_explorer(&[snapshot], &TestExplorerFilter::default());
        let case = find_node(&projection.snapshot.roots, "messageIsAvailable")
            .expect("selected Java case");
        let selection = projection
            .selection(&case.id)
            .expect("Java selection")
            .clone();
        let (container, cx) = cx.add_window_view(|window, cx| {
            workspace::MultiWorkspace::test_new(project.clone(), window, cx)
        });
        let workspace = container.read_with(cx, |container, _| container.workspace().clone());
        cx.update(|_, cx| {
            if cx.has_global::<project::trusted_worktrees::TrustedWorktrees>() {
                cx.remove_global::<project::trusted_worktrees::TrustedWorktrees>();
            }
        });
        cx.update(|window, cx| {
            delegate
                .run(&selection, &workspace.downgrade(), window, cx)
                .expect("dispatch selected Java Task")
        });
        cx.run_until_parked();
        let structured = project.read_with(cx, |project, _| {
            project.structured_execution_store().clone()
        });
        let run = structured
            .read_with(cx, |store, _| {
                store
                    .state()
                    .provider(&StructuredProviderId(JVM_JUNIT_PROVIDER_ID.to_string()))
                    .and_then(|snapshot| snapshot.current_run.clone())
            })
            .expect("selected Java Task reached structured Tests");
        assert_eq!(run.scope_node_ids, vec![selection.node_id.clone()]);
        assert_eq!(
            run.phase,
            project::structured_execution::StructuredRunPhase::SpawnError
        );
        assert!(
            run.diagnostic
                .as_deref()
                .is_some_and(|message| message.contains("Trust the project"))
        );
        let spawn_error_selection = TestExplorerSelection {
            run_id: Some(run.id),
            ..selection.clone()
        };
        assert!(
            cx.update(|window, cx| {
                delegate.rerun_failed(
                    std::slice::from_ref(&spawn_error_selection),
                    &workspace.downgrade(),
                    window,
                    cx,
                )
            })
            .is_err(),
            "a Task spawn error is not a completed Java test failure"
        );
        let failed_run_id = StructuredRunId("java-failed-before-rerun".to_string());
        structured.update(cx, |store, cx| {
            let provider_id = StructuredProviderId(JVM_JUNIT_PROVIDER_ID.to_string());
            let project_generation = store.state().project_generation();
            store
                .state_mut()
                .begin_run(
                    project_generation,
                    &provider_id,
                    StructuredRun::new(
                        failed_run_id.clone(),
                        selection.discovery_generation,
                        vec![selection.node_id.clone()],
                    ),
                )
                .expect("current failed Java run");
            store
                .state_mut()
                .apply_event(
                    project_generation,
                    &provider_id,
                    &failed_run_id,
                    StructuredExecutionEvent {
                        sequence: 0,
                        node_id: selection.node_id.clone(),
                        state: StructuredNodeState::Failed,
                        duration_millis: None,
                        message: Some("fixture failure".to_string()),
                        location: None,
                    },
                    None,
                )
                .expect("failed Java case");
            store
                .state_mut()
                .set_run_phase(
                    project_generation,
                    &provider_id,
                    &failed_run_id,
                    project::structured_execution::StructuredRunPhase::Completed,
                    None,
                )
                .expect("completed failed Java run");
            cx.emit(StructuredExecutionStoreEvent::Changed(provider_id));
        });
        let failed_selection = TestExplorerSelection {
            run_id: Some(failed_run_id),
            ..selection
        };
        cx.update(|window, cx| {
            delegate
                .rerun_failed(
                    std::slice::from_ref(&failed_selection),
                    &workspace.downgrade(),
                    window,
                    cx,
                )
                .expect("rerun the current failed Java case")
        });
        cx.run_until_parked();
        let rerun = structured
            .read_with(cx, |store, _| {
                store
                    .state()
                    .provider(&StructuredProviderId(JVM_JUNIT_PROVIDER_ID.to_string()))
                    .and_then(|snapshot| snapshot.current_run.clone())
            })
            .expect("Java rerun reached structured Tests");
        assert_ne!(
            &rerun.id,
            failed_selection.run_id.as_ref().expect("failed run id")
        );
        assert_eq!(
            rerun.phase,
            project::structured_execution::StructuredRunPhase::SpawnError
        );
        assert!(
            cx.update(|window, cx| {
                delegate.rerun_failed(
                    std::slice::from_ref(&failed_selection),
                    &workspace.downgrade(),
                    window,
                    cx,
                )
            })
            .is_err()
        );
        cx.deactivate_window();
        container.update_in(cx, |_, window, _| window.remove_window());
    }

    #[cfg(all(feature = "jvm-test-actions", unix))]
    #[gpui::test]
    #[ignore = "requires a JDK and populated Maven wrapper and dependency caches"]
    async fn selected_java_maven_case_completes_through_tests_panel(cx: &mut TestAppContext) {
        selected_java_case_through_tests_panel(cx, JvmTestBuildTool::Maven, JavaTestOutcome::Pass)
            .await;
    }

    #[cfg(all(feature = "jvm-test-actions", unix))]
    #[gpui::test]
    #[ignore = "requires a JDK and populated Maven wrapper and dependency caches"]
    async fn failed_java_maven_case_reruns_through_tests_panel(cx: &mut TestAppContext) {
        selected_java_case_through_tests_panel(
            cx,
            JvmTestBuildTool::Maven,
            JavaTestOutcome::FailAndRerun,
        )
        .await;
    }

    #[cfg(all(feature = "jvm-test-actions", unix))]
    #[gpui::test]
    #[ignore = "requires a JDK and populated Gradle wrapper and dependency caches"]
    async fn selected_java_gradle_case_completes_through_tests_panel(cx: &mut TestAppContext) {
        selected_java_case_through_tests_panel(cx, JvmTestBuildTool::Gradle, JavaTestOutcome::Pass)
            .await;
    }

    #[cfg(all(feature = "jvm-test-actions", unix))]
    #[gpui::test]
    #[ignore = "requires GROOVY_JDTLS_LAUNCHER, GROOVY_KOTLIN_LSP, GROOVY_LSP_JAVA_HOME, GROOVY_BUILD_JAVA_HOME and cached Gradle dependencies"]
    async fn cached_gradle_providers_build_dependencies_and_tests_share_one_project(
        cx: &mut TestAppContext,
    ) {
        selected_java_case_through_tests_panel(
            cx,
            JvmTestBuildTool::Gradle,
            JavaTestOutcome::CombinedGradleWorkflow,
        )
        .await;
    }

    #[cfg(all(feature = "jvm-test-actions", unix))]
    #[gpui::test]
    async fn cancelling_a_selected_java_child_closes_the_tests_run(cx: &mut TestAppContext) {
        selected_java_case_through_tests_panel(
            cx,
            JvmTestBuildTool::Maven,
            JavaTestOutcome::Cancel,
        )
        .await;
    }

    #[cfg(all(feature = "jvm-test-actions", unix))]
    #[gpui::test]
    async fn restricting_trust_stops_a_selected_java_child(cx: &mut TestAppContext) {
        selected_java_case_through_tests_panel(
            cx,
            JvmTestBuildTool::Maven,
            JavaTestOutcome::RestrictTrust,
        )
        .await;
    }

    #[cfg(all(feature = "jvm-test-actions", unix))]
    #[gpui::test]
    #[ignore = "requires a JDK and populated Maven wrapper and dependency caches"]
    async fn restricting_trust_stops_a_selected_junit_jvm(cx: &mut TestAppContext) {
        selected_java_case_through_tests_panel(
            cx,
            JvmTestBuildTool::Maven,
            JavaTestOutcome::RestrictJavaProcess,
        )
        .await;
    }

    #[cfg(all(feature = "jvm-test-actions", unix))]
    #[gpui::test]
    #[ignore = "requires a JDK and populated Gradle wrapper and dependency caches"]
    async fn restricting_trust_stops_a_selected_gradle_junit_jvm(cx: &mut TestAppContext) {
        selected_java_case_through_tests_panel(
            cx,
            JvmTestBuildTool::Gradle,
            JavaTestOutcome::RestrictJavaProcess,
        )
        .await;
    }

    #[cfg(all(feature = "jvm-test-actions", unix))]
    #[gpui::test]
    #[ignore = "requires a JDK and a populated Maven wrapper cache"]
    async fn missing_offline_maven_cache_keeps_selected_case_unverified(cx: &mut TestAppContext) {
        selected_java_case_through_tests_panel(
            cx,
            JvmTestBuildTool::Maven,
            JavaTestOutcome::MissingMavenCache,
        )
        .await;
    }

    #[cfg(all(feature = "jvm-test-actions", windows))]
    #[gpui::test]
    async fn selected_java_windows_maven_case_uses_tests_panel_and_task_terminal(
        cx: &mut TestAppContext,
    ) {
        selected_java_case_through_tests_panel(cx, JvmTestBuildTool::Maven, JavaTestOutcome::Pass)
            .await;
    }

    #[cfg(all(feature = "jvm-test-actions", target_os = "macos"))]
    #[gpui::test]
    #[ignore = "requires current ZED_COPY_REMOTE_SERVER, verified GROOVY_SSH_* loopback fixture, host Java 6.8.27 extension and cached Maven/JDK"]
    async fn restricting_trust_stops_a_selected_junit_over_ssh(cx: &mut TestAppContext) {
        selected_java_case_through_tests_panel_with_connection(
            cx,
            JvmTestBuildTool::Maven,
            JavaTestOutcome::RestrictJavaProcess,
            true,
        )
        .await;
    }

    #[cfg(all(feature = "jvm-test-actions", target_os = "macos"))]
    #[gpui::test]
    #[ignore = "requires current ZED_COPY_REMOTE_SERVER, verified GROOVY_SSH_* loopback fixture, host Java 6.8.27 extension and cached Maven/JDK"]
    async fn cancelling_stops_a_selected_junit_over_ssh(cx: &mut TestAppContext) {
        selected_java_case_through_tests_panel_with_connection(
            cx,
            JvmTestBuildTool::Maven,
            JavaTestOutcome::CancelJavaProcess,
            true,
        )
        .await;
    }

    #[cfg(all(feature = "jvm-test-actions", target_os = "macos"))]
    #[gpui::test]
    #[ignore = "requires current ZED_COPY_REMOTE_SERVER, verified GROOVY_SSH_* loopback fixture, host Java 6.8.27 extension and cached Maven/JDK"]
    async fn disconnecting_stops_a_selected_junit_over_ssh(cx: &mut TestAppContext) {
        selected_java_case_through_tests_panel_with_connection(
            cx,
            JvmTestBuildTool::Maven,
            JavaTestOutcome::DisconnectJavaProcess,
            true,
        )
        .await;
    }

    fn path(value: &str) -> ProjectPath {
        ProjectPath {
            worktree_id: WorktreeId::from_usize(1),
            path: Arc::from(
                RelPath::from_unix_str(value).expect("fixture path should be relative"),
            ),
        }
    }

    fn fake_provider(status: StructuredProviderStatus) -> StructuredProviderSnapshot {
        let provider_id = StructuredProviderId("web-tests".to_string());
        let root_id = StructuredNodeId("web-root".to_string());
        let suite_id = StructuredNodeId("browser-suite".to_string());
        let passed_id = StructuredNodeId("renders-home".to_string());
        let failed_id = StructuredNodeId("renders-login".to_string());
        let mut run = StructuredRun::new(
            StructuredRunId("run-1".to_string()),
            DiscoveryGeneration(3),
            vec![suite_id.clone()],
        );
        run.events = vec![
            StructuredExecutionEvent {
                sequence: 0,
                node_id: passed_id.clone(),
                state: StructuredNodeState::Passed,
                duration_millis: Some(4),
                message: None,
                location: Some(path("web/home.test.js")),
            },
            StructuredExecutionEvent {
                sequence: 1,
                node_id: failed_id.clone(),
                state: StructuredNodeState::Failed,
                duration_millis: Some(9),
                message: Some("expected dashboard".to_string()),
                location: Some(path("web/login.test.js")),
            },
        ];
        run.summary = StructuredExecutionSummary {
            total: 2,
            passed: 1,
            failed: 1,
            ..Default::default()
        };
        StructuredProviderSnapshot {
            provider_id,
            discovery_generation: DiscoveryGeneration(3),
            status,
            nodes: vec![
                StructuredNode {
                    experimental_feature: None,
                    experimental_feature_enabled: None,
                    debug_capability: None,
                    id: root_id.clone(),
                    parent_id: None,
                    label: "Web Tests".to_string(),
                    kind: StructuredNodeKind::Provider,
                    path: None,
                },
                StructuredNode {
                    experimental_feature: None,
                    experimental_feature_enabled: None,
                    debug_capability: None,
                    id: suite_id.clone(),
                    parent_id: Some(root_id),
                    label: "Browser".to_string(),
                    kind: StructuredNodeKind::Suite,
                    path: Some(path("web")),
                },
                StructuredNode {
                    experimental_feature: None,
                    experimental_feature_enabled: None,
                    debug_capability: Some(StructuredDebugCapability::Supported),
                    id: passed_id,
                    parent_id: Some(suite_id.clone()),
                    label: "renders home".to_string(),
                    kind: StructuredNodeKind::Case,
                    path: Some(path("web/home.test.js")),
                },
                StructuredNode {
                    experimental_feature: None,
                    experimental_feature_enabled: None,
                    debug_capability: Some(StructuredDebugCapability::Supported),
                    id: failed_id,
                    parent_id: Some(suite_id),
                    label: "renders login".to_string(),
                    kind: StructuredNodeKind::Case,
                    path: Some(path("web/login.test.js")),
                },
            ],
            partial: status == StructuredProviderStatus::Partial,
            diagnostic: (status != StructuredProviderStatus::Current)
                .then(|| provider_status_default_message(status).to_string()),
            current_run: None,
            last_complete_run: Some(run),
            completed_runs: VecDeque::new(),
        }
    }

    struct QueuedDiscoveryDelegate(
        std::sync::Mutex<VecDeque<futures::channel::oneshot::Receiver<StructuredProviderSnapshot>>>,
    );

    impl TestExplorerActionDelegate for QueuedDiscoveryDelegate {
        fn discover(
            &self,
            _: &Entity<Project>,
            cx: &mut App,
        ) -> Vec<(
            StructuredProviderId,
            gpui::Task<anyhow::Result<StructuredProviderSnapshot>>,
        )> {
            let receiver = self
                .0
                .lock()
                .expect("discovery queue lock")
                .pop_front()
                .expect("queued discovery response");
            vec![(
                StructuredProviderId("web-tests".into()),
                cx.background_spawn(async move { Ok(receiver.await?) }),
            )]
        }

        fn run(
            &self,
            _: &TestExplorerSelection,
            _: &WeakEntity<Workspace>,
            _: &mut Window,
            _: &mut App,
        ) -> anyhow::Result<()> {
            anyhow::bail!("test delegate does not run tests")
        }

        fn cancel(
            &self,
            _: &TestExplorerSelection,
            _: &WeakEntity<Workspace>,
            _: &mut Window,
            _: &mut App,
        ) -> anyhow::Result<()> {
            anyhow::bail!("test delegate has no active run")
        }

        fn debug(
            &self,
            _: &TestExplorerSelection,
            _: &WeakEntity<Workspace>,
            _: &mut Window,
            _: &mut App,
        ) -> anyhow::Result<()> {
            anyhow::bail!("test delegate does not debug tests")
        }

        fn rerun_failed(
            &self,
            _: &[TestExplorerSelection],
            _: &WeakEntity<Workspace>,
            _: &mut Window,
            _: &mut App,
        ) -> anyhow::Result<()> {
            anyhow::bail!("test delegate has no completed run")
        }

        fn reveal_terminal(
            &self,
            _: &TestExplorerSelection,
            _: &WeakEntity<Workspace>,
            _: &mut Window,
            _: &mut App,
        ) -> anyhow::Result<()> {
            anyhow::bail!("test delegate has no terminal")
        }
    }

    #[test]
    fn test_explorer_aggregate_phases_leave_child_results_unknown() {
        use project::structured_execution::StructuredRunPhase;

        let mut provider = fake_provider(StructuredProviderStatus::Current);
        let target_id = StructuredNodeId("aggregate-target".into());
        let module_id = StructuredNodeId("aggregate-module".into());
        if let Some(previous) = provider.last_complete_run.as_mut() {
            previous.phase = StructuredRunPhase::Completed;
        }
        let (mut parent_nodes, case_nodes): (Vec<_>, Vec<_>) = provider
            .nodes
            .into_iter()
            .partition(|node| node.kind != StructuredNodeKind::Case);
        parent_nodes.extend([
            StructuredNode {
                experimental_feature: None,
                experimental_feature_enabled: None,
                debug_capability: None,
                id: target_id.clone(),
                parent_id: Some(StructuredNodeId("browser-suite".into())),
                label: "Browser target".into(),
                kind: StructuredNodeKind::Group,
                path: None,
            },
            StructuredNode {
                experimental_feature: None,
                experimental_feature_enabled: None,
                debug_capability: None,
                id: module_id.clone(),
                parent_id: Some(target_id.clone()),
                label: "Browser module".into(),
                kind: StructuredNodeKind::Group,
                path: None,
            },
        ]);
        parent_nodes.extend(case_nodes.into_iter().map(|mut node| {
            node.parent_id = Some(module_id.clone());
            node
        }));
        provider.nodes = parent_nodes;
        for scope_node_ids in [
            vec![StructuredNodeId("browser-suite".into())],
            vec![target_id],
            vec![module_id],
            vec![
                StructuredNodeId("renders-home".into()),
                StructuredNodeId("renders-login".into()),
            ],
        ] {
            for (phase, diagnostic, expected_label) in [
                (StructuredRunPhase::Queued, None, "task queued"),
                (StructuredRunPhase::Running, None, "task running"),
                (StructuredRunPhase::Completed, None, "task completed"),
                (
                    StructuredRunPhase::Completed,
                    Some("Task exited with status 7"),
                    "task completed; Task exited with status 7",
                ),
                (
                    StructuredRunPhase::SpawnError,
                    Some("Task could not start"),
                    "task failed to start; Task could not start",
                ),
                (
                    StructuredRunPhase::Cancelled,
                    Some("Termination is not confirmed"),
                    "task cancelled; Termination is not confirmed",
                ),
            ] {
                let mut provider = provider.clone();
                let mut run = StructuredRun::new(
                    StructuredRunId("aggregate-current".into()),
                    provider.discovery_generation,
                    scope_node_ids.clone(),
                );
                run.phase = phase;
                run.diagnostic = diagnostic.map(str::to_owned);
                provider.current_run = Some(run);
                let projection = project_test_explorer(&[provider], &TestExplorerFilter::default());
                let root = find_node(&projection.snapshot.roots, "Web Tests")
                    .expect("aggregate provider should be visible");
                let expected = format!("{expected_label}; per-test results unavailable");
                assert_eq!(root.secondary_label.as_deref(), Some(expected.as_str()));
                assert_eq!(root.accessibility_label, format!("Web Tests, {expected}"));
                assert_eq!(projection.summary, StructuredExecutionSummary::default());
                assert!(projection.failed.is_empty());
                for label in ["renders home", "renders login"] {
                    let child = find_node(&projection.snapshot.roots, label)
                        .expect("unknown child should remain visible in All");
                    assert!(child.secondary_label.is_none());
                    assert_eq!(child.icon, Some(IconName::Circle));
                    assert_eq!(child.accessibility_label, label);
                    let selection = projection
                        .selections
                        .get(&child.id)
                        .expect("child selection should retain the aggregate run");
                    assert_eq!(
                        selection
                            .run_id
                            .as_ref()
                            .map(|identifier| identifier.0.as_str()),
                        Some("aggregate-current")
                    );
                }
            }
        }
    }

    #[test]
    fn test_explorer_aggregate_summary_and_filters_do_not_infer_case_outcomes() {
        let ordinary = fake_provider(StructuredProviderStatus::Current);
        let ordinary_projection = project_test_explorer(
            std::slice::from_ref(&ordinary),
            &TestExplorerFilter::default(),
        );
        assert_eq!(
            ordinary_projection.summary_label(),
            "2 total · 1 passed · 1 failed · 0 skipped"
        );
        for (phase, matching_filter) in [
            (StructuredRunPhase::Queued, Some(TestStatusFilter::Running)),
            (StructuredRunPhase::Running, Some(TestStatusFilter::Running)),
            (
                StructuredRunPhase::SpawnError,
                Some(TestStatusFilter::Failed),
            ),
            (
                StructuredRunPhase::Cancelled,
                Some(TestStatusFilter::Cancelled),
            ),
            (StructuredRunPhase::Completed, None),
        ] {
            let mut aggregate = ordinary.clone();
            aggregate.provider_id = StructuredProviderId("aggregate-web-tests".into());
            let mut run = StructuredRun::new(
                StructuredRunId("aggregate-only".into()),
                aggregate.discovery_generation,
                vec![StructuredNodeId("browser-suite".into())],
            );
            run.phase = phase;
            run.diagnostic = Some("Task exited with status 7".into());
            aggregate.current_run = Some(run.clone());
            let projection =
                project_test_explorer(&[aggregate.clone()], &TestExplorerFilter::default());
            assert_eq!(
                projection.summary_label(),
                "Per-test results unavailable; task status is shown on provider rows"
            );
            for filter in [
                TestStatusFilter::Failed,
                TestStatusFilter::Passed,
                TestStatusFilter::Skipped,
                TestStatusFilter::Running,
                TestStatusFilter::Cancelled,
            ] {
                let projected = project_test_explorer(
                    &[aggregate.clone()],
                    &TestExplorerFilter {
                        query: String::new(),
                        status: filter,
                    },
                );
                assert_eq!(
                    projected.snapshot.roots.is_empty(),
                    Some(filter) != matching_filter
                );
                assert!(projected.failed.is_empty());
                assert!(find_node(&projected.snapshot.roots, "renders login").is_none());
            }
            let mixed = project_test_explorer(
                &[ordinary.clone(), aggregate.clone()],
                &TestExplorerFilter::default(),
            );
            assert_eq!(mixed.summary, ordinary_projection.summary);
            assert_eq!(mixed.failed.len(), 1);
            assert_eq!(
                mixed.summary_label(),
                "2 total · 1 passed · 1 failed · 0 skipped; some tasks have no per-test results"
            );
            if phase == StructuredRunPhase::Completed {
                aggregate.current_run = None;
                aggregate.last_complete_run = Some(run);
                let historical =
                    project_test_explorer(&[aggregate], &TestExplorerFilter::default());
                assert_eq!(historical.summary, StructuredExecutionSummary::default());
                assert!(
                    historical
                        .summary_label()
                        .starts_with("Per-test results unavailable")
                );
                assert!(historical.selections.values().all(|selection| {
                    selection.run_id.as_ref().map(|id| id.0.as_str()) == Some("aggregate-only")
                }));
            }
        }
    }

    struct DebugCapabilityDelegate(Arc<std::sync::atomic::AtomicUsize>);

    impl TestExplorerActionDelegate for DebugCapabilityDelegate {
        fn run(
            &self,
            _: &TestExplorerSelection,
            _: &WeakEntity<Workspace>,
            _: &mut Window,
            _: &mut App,
        ) -> anyhow::Result<()> {
            anyhow::bail!("unexpected run in Debug capability fixture")
        }
        fn cancel(
            &self,
            _: &TestExplorerSelection,
            _: &WeakEntity<Workspace>,
            _: &mut Window,
            _: &mut App,
        ) -> anyhow::Result<()> {
            anyhow::bail!("unexpected cancel in Debug capability fixture")
        }
        fn rerun_failed(
            &self,
            _: &[TestExplorerSelection],
            _: &WeakEntity<Workspace>,
            _: &mut Window,
            _: &mut App,
        ) -> anyhow::Result<()> {
            anyhow::bail!("unexpected rerun_failed in Debug capability fixture")
        }
        fn reveal_terminal(
            &self,
            _: &TestExplorerSelection,
            _: &WeakEntity<Workspace>,
            _: &mut Window,
            _: &mut App,
        ) -> anyhow::Result<()> {
            anyhow::bail!("unexpected reveal_terminal in Debug capability fixture")
        }
        fn debug(
            &self,
            _: &TestExplorerSelection,
            _: &WeakEntity<Workspace>,
            _: &mut Window,
            _: &mut App,
        ) -> anyhow::Result<()> {
            self.0.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            Ok(())
        }
    }

    #[gpui::test]
    fn experimental_test_actions_follow_local_opt_in_without_disabling_ordinary_cases(
        cx: &mut App,
    ) {
        use feature_flags::{
            FeatureFlag, FeatureFlagsSettings, RustExactDoctestActionsExperimentalFeatureFlag,
        };
        let settings_store = SettingsStore::test(cx);
        cx.set_global(settings_store);
        SettingsStore::update_global(cx, |store, _| {
            store.register_setting::<FeatureFlagsSettings>()
        });
        FeatureFlagStore::init(cx);
        let feature = RustExactDoctestActionsExperimentalFeatureFlag::NAME;
        cx.update_flags(true, vec![feature.into()]);
        let mut provider = fake_provider(StructuredProviderStatus::Current);
        let case = provider
            .nodes
            .iter_mut()
            .find(|node| node.id.0 == "renders-login")
            .expect("failed case");
        case.experimental_feature = Some(feature.into());
        case.experimental_feature_enabled = Some(true);
        let projection = project_test_explorer(&[provider], &TestExplorerFilter::default());
        let experimental = tree_id(
            &StructuredProviderId("web-tests".into()),
            &StructuredNodeId("renders-login".into()),
        );
        let ordinary = tree_id(
            &StructuredProviderId("web-tests".into()),
            &StructuredNodeId("renders-home".into()),
        );
        for value in [None, Some("on"), Some("off"), Some("invalid")] {
            SettingsStore::update_global(cx, |store, cx| {
                store.update_user_settings(cx, |content| {
                    let flags = content.feature_flags.get_or_insert_default();
                    match value {
                        Some(value) => {
                            flags.insert(feature.into(), value.into());
                        }
                        None => {
                            flags.remove(feature);
                        }
                    }
                })
            });
            let enabled = value == Some("on");
            assert_eq!(projection.can_run_with_flags(&experimental, cx), enabled);
            assert_eq!(projection.can_debug_with_flags(&experimental, cx), enabled);
            assert_eq!(projection.enabled_failed(cx).len(), usize::from(enabled));
            assert!(projection.can_run_with_flags(&ordinary, cx));
            assert!(projection.can_debug_with_flags(&ordinary, cx));
        }
    }

    #[gpui::test]
    fn experimental_test_controls_require_host_and_client_opt_in(cx: &mut App) {
        use feature_flags::{FeatureFlag, RustExactDoctestActionsExperimentalFeatureFlag};
        let feature = RustExactDoctestActionsExperimentalFeatureFlag::NAME;
        project::task_store::set_experimental_feature_for_test(feature, true, cx);
        let selected = tree_id(
            &StructuredProviderId("web-tests".into()),
            &StructuredNodeId("renders-login".into()),
        );
        let ordinary = tree_id(
            &StructuredProviderId("web-tests".into()),
            &StructuredNodeId("renders-home".into()),
        );
        for host_enabled in [None, Some(false), Some(true)] {
            let mut provider = fake_provider(StructuredProviderStatus::Current);
            let node = provider
                .nodes
                .iter_mut()
                .find(|node| node.id.0 == "renders-login")
                .expect("exact case");
            node.experimental_feature = Some(feature.into());
            node.experimental_feature_enabled = host_enabled;
            let projection = project_test_explorer(&[provider], &TestExplorerFilter::default());
            assert_eq!(
                projection.can_run_with_flags(&selected, cx),
                host_enabled == Some(true)
            );
            assert_eq!(
                projection.can_debug_with_flags(&selected, cx),
                host_enabled == Some(true)
            );
            assert_eq!(
                projection.enabled_failed(cx).len(),
                usize::from(host_enabled == Some(true))
            );
            assert!(projection.can_run_with_flags(&ordinary, cx));
            assert!(projection.can_debug_with_flags(&ordinary, cx));
        }
    }

    #[gpui::test]
    fn unknown_experimental_test_requirement_does_not_authorize_actions(cx: &mut App) {
        let mut provider = fake_provider(StructuredProviderStatus::Current);
        provider
            .nodes
            .iter_mut()
            .find(|node| node.id.0 == "renders-login")
            .expect("failed case")
            .experimental_feature = Some("unknown-experimental-action".into());
        let projection = project_test_explorer(&[provider], &TestExplorerFilter::default());
        let selected = tree_id(
            &StructuredProviderId("web-tests".into()),
            &StructuredNodeId("renders-login".into()),
        );
        assert!(!projection.can_run_with_flags(&selected, cx));
        assert!(!projection.can_debug_with_flags(&selected, cx));
        assert!(projection.enabled_failed(cx).is_empty());
    }

    #[gpui::test]
    async fn experimental_test_controls_and_direct_handlers_refresh_on_revocation(
        cx: &mut TestAppContext,
    ) {
        use feature_flags::{
            FeatureFlag, FeatureFlagsSettings, RustExactDoctestActionsExperimentalFeatureFlag,
        };
        crate::tests::init_test(cx);
        cx.update(|cx| {
            SettingsStore::update_global(cx, |store, _| {
                store.register_setting::<FeatureFlagsSettings>()
            });
            FeatureFlagStore::init(cx);
        });
        let feature = RustExactDoctestActionsExperimentalFeatureFlag::NAME;
        let project = Project::test(fs::FakeFs::new(cx.executor()), [], cx).await;
        let mut provider = fake_provider(StructuredProviderStatus::Current);
        let case = provider
            .nodes
            .iter_mut()
            .find(|node| node.id.0 == "renders-login")
            .expect("failed case");
        case.experimental_feature = Some(feature.into());
        case.experimental_feature_enabled = Some(true);
        let store = project.read_with(cx, |project, _| {
            project.structured_execution_store().clone()
        });
        store.update(cx, |store, _| {
            let generation = store.state().project_generation();
            store
                .state_mut()
                .apply_discovery(generation, provider, None)
                .expect("discovery");
        });
        let (container, cx) = cx
            .add_window_view(|window, cx| workspace::MultiWorkspace::test_new(project, window, cx));
        let workspace = container.read_with(cx, |container, _| container.workspace().clone());
        let async_cx = cx.update(|window, cx| window.to_async(cx));
        let panel = TestsPanel::load(workspace.downgrade(), async_cx)
            .await
            .expect("Tests panel");
        workspace.update_in(cx, |workspace, window, cx| {
            workspace.add_panel(panel.clone(), window, cx);
            workspace.focus_panel::<TestsPanel>(window, cx);
            window.activate_a11y_for_test();
        });
        let debug_calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        panel.update(cx, |panel, cx| {
            panel.set_delegate(Arc::new(DebugCapabilityDelegate(debug_calls.clone())));
            panel.active = true;
            panel.refresh(cx);
        });
        cx.run_until_parked();
        panel.update(cx, |panel, cx| {
            panel.host.expand_all();
            panel.host.select(tree_id(
                &StructuredProviderId("web-tests".into()),
                &StructuredNodeId("renders-login".into()),
            ));
            cx.notify();
        });
        for (value, expected_calls) in [("off", 0), ("on", 1), ("off", 1)] {
            cx.update(|_, cx| {
                SettingsStore::update_global(cx, |store, cx| {
                    store.update_user_settings(cx, |content| {
                        content
                            .feature_flags
                            .get_or_insert_default()
                            .insert(feature.into(), value.into());
                    })
                })
            });
            cx.run_until_parked();
            cx.update(|window, cx| {
                window.draw(cx).clear(cx);
                let data: serde_json::Value =
                    serde_json::from_str(&window.debug_a11y_tree_json().expect("Tests tree"))
                        .expect("valid tree");
                for label in [
                    "Run Selected Tests",
                    "Debug Selected Test",
                    "Rerun Failed Tests",
                ] {
                    let control = data["nodes"]
                        .as_object()
                        .expect("nodes")
                        .values()
                        .find(|node| node["aria"]["label"] == label)
                        .expect("control");
                    assert_eq!(control["aria"]["disabled"], value != "on", "{control}");
                }
                window.dispatch_action(DebugSelectedTests.boxed_clone(), cx);
            });
            cx.run_until_parked();
            assert_eq!(
                debug_calls.load(std::sync::atomic::Ordering::SeqCst),
                expected_calls
            );
            panel.update_in(cx, |panel, window, cx| {
                if value == "off" {
                    assert!(
                        panel
                            .action_notice
                            .as_deref()
                            .is_some_and(|notice| notice.contains(feature))
                    );
                    panel.run_selected(window, cx);
                    assert!(
                        panel
                            .action_notice
                            .as_deref()
                            .is_some_and(|notice| notice.contains(feature))
                    );
                    panel.rerun_failed(window, cx);
                    assert_eq!(
                        panel.action_notice.as_deref(),
                        Some("No enabled failed test actions are available to rerun")
                    );
                }
                let selected = panel.host.selected().expect("selection").clone();
                panel.deploy_context_menu(Point::default(), selected, window, cx);
                assert!(panel.context_menu.is_some());
                window.focus(&panel.focus_handle, cx);
            });
        }
        cx.update(|_, cx| {
            SettingsStore::update_global(cx, |store, cx| {
                store.update_user_settings(cx, |content| {
                    content
                        .feature_flags
                        .get_or_insert_default()
                        .insert(feature.into(), "on".into());
                })
            })
        });
        cx.run_until_parked();
        assert!(panel.read_with(cx, |panel, _| panel.context_menu.is_none()));
    }

    #[gpui::test]
    async fn test_explorer_debug_requires_provider_capability(cx: &mut TestAppContext) {
        crate::tests::init_test(cx);
        let project = Project::test(fs::FakeFs::new(cx.executor()), [], cx).await;
        let mut provider = fake_provider(StructuredProviderStatus::Current);
        for node in &mut provider.nodes {
            node.debug_capability = None;
        }
        let provider_id = provider.provider_id.clone();
        let store = project.read_with(cx, |project, _| {
            project.structured_execution_store().clone()
        });
        store.update(cx, |store, _| {
            let generation = store.state().project_generation();
            store
                .state_mut()
                .apply_discovery(generation, provider, None)
                .expect("discovery");
        });
        let (container, cx) = cx
            .add_window_view(|window, cx| workspace::MultiWorkspace::test_new(project, window, cx));
        let workspace = container.read_with(cx, |container, _| container.workspace().clone());
        let async_cx = cx.update(|window, cx| window.to_async(cx));
        let panel = TestsPanel::load(workspace.downgrade(), async_cx)
            .await
            .expect("Tests panel");
        workspace.update_in(cx, |workspace, window, cx| {
            workspace.add_panel(panel.clone(), window, cx);
            workspace.focus_panel::<TestsPanel>(window, cx);
            window.activate_a11y_for_test();
        });
        cx.run_until_parked();
        let debug_calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        panel.update(cx, |panel, cx| {
            panel.set_delegate(Arc::new(DebugCapabilityDelegate(debug_calls.clone())));
            panel.active = true;
            panel.refresh(cx);
            panel.host.expand_all();
            panel.host.select(tree_id(
                &provider_id,
                &StructuredNodeId("renders-home".into()),
            ));
            cx.notify();
        });
        cx.run_until_parked();
        let mut expected_debug_calls = 0;
        for (index, (capability, status, reason)) in [
            (
                None,
                StructuredProviderStatus::Current,
                Some("This test provider has not reported Debug support"),
            ),
            (
                Some(StructuredDebugCapability::Supported),
                StructuredProviderStatus::Current,
                None,
            ),
            (
                Some(StructuredDebugCapability::Unsupported(
                    "Cargo DAP does not support debugging doctests".into(),
                )),
                StructuredProviderStatus::Current,
                Some("Cargo DAP does not support debugging doctests"),
            ),
            (
                Some(StructuredDebugCapability::Unsupported(
                    "Cargo DAP does not support debugging an individual benchmark".into(),
                )),
                StructuredProviderStatus::Current,
                Some("Cargo DAP does not support debugging an individual benchmark"),
            ),
            (
                Some(StructuredDebugCapability::Unsupported(
                    "The selected Rust test has no debuggable harness".into(),
                )),
                StructuredProviderStatus::Current,
                Some("The selected Rust test has no debuggable harness"),
            ),
            (
                Some(StructuredDebugCapability::Supported),
                StructuredProviderStatus::Stale,
                Some("Refresh test discovery before debugging"),
            ),
            (
                Some(StructuredDebugCapability::Supported),
                StructuredProviderStatus::Partial,
                None,
            ),
        ]
        .into_iter()
        .enumerate()
        {
            let mut provider = fake_provider(status);
            provider.discovery_generation = DiscoveryGeneration(4 + index as u64);
            provider.last_complete_run = None;
            let case = provider
                .nodes
                .iter_mut()
                .find(|node| node.id.0 == "renders-home")
                .expect("case");
            case.debug_capability = capability;
            store.update(cx, |store, cx| {
                let generation = store.state().project_generation();
                store
                    .state_mut()
                    .apply_discovery(generation, provider, None)
                    .expect("updated discovery");
                cx.emit(StructuredExecutionStoreEvent::Changed(provider_id.clone()));
            });
            cx.run_until_parked();
            for _ in 0..2 {
                cx.update(|window, cx| {
                    window.draw(cx).clear(cx);
                    assert!(panel.read(cx).focus_handle.is_focused(window));
                    let data = window.debug_a11y_tree_json().expect("Tests tree");
                    let data: serde_json::Value = serde_json::from_str(&data).expect("valid tree");
                    let control = data["nodes"]
                        .as_object()
                        .expect("nodes")
                        .values()
                        .find(|node| node["aria"]["label"] == "Debug Selected Test")
                        .expect("Debug control");
                    assert_eq!(control["aria"]["disabled"], reason.is_some(), "{control}");
                    assert_eq!(control["aria"]["description"].as_str(), reason, "{control}");
                });
            }
            cx.update(|window, cx| window.dispatch_action(DebugSelectedTests.boxed_clone(), cx));
            cx.run_until_parked();
            panel.read_with(cx, |panel, _| {
                assert_eq!(panel.action_notice.as_deref(), reason);
            });
            if reason.is_none() {
                expected_debug_calls += 1;
            }
            assert_eq!(
                debug_calls.load(std::sync::atomic::Ordering::SeqCst),
                expected_debug_calls
            );
            panel.update_in(cx, |panel, window, cx| {
                let selected = panel.host.selected().expect("selected case").clone();
                panel.deploy_context_menu(Point::default(), selected, window, cx);
                let (menu, _, _) = panel.context_menu.as_ref().expect("context menu");
                let expected_label = reason
                    .map(|reason| format!("Debug: {reason}"))
                    .unwrap_or_else(|| "Debug".into());
                let entries = menu.read(cx).accessibility_contracts();
                let entry = entries
                    .iter()
                    .find(|entry| entry.label.as_ref() == expected_label)
                    .expect("Debug entry and reason");
                assert_eq!(entry.disabled, reason.is_some());
                panel.context_menu.take();
                window.focus(&panel.focus_handle, cx);
                cx.notify();
            });
            cx.run_until_parked();
        }
    }

    #[gpui::test]
    async fn test_explorer_aggregate_store_updates_reach_loaded_panel_and_paint(
        cx: &mut TestAppContext,
    ) {
        crate::tests::init_test(cx);
        let project = Project::test(fs::FakeFs::new(cx.executor()), [], cx).await;
        let mut provider = fake_provider(StructuredProviderStatus::Current);
        for node in &mut provider.nodes {
            node.path = None;
        }
        if let Some(previous) = provider.last_complete_run.as_mut() {
            previous.phase = StructuredRunPhase::Completed;
        }
        let provider_id = provider.provider_id.clone();
        let discovery_generation = provider.discovery_generation;
        let store = project.read_with(cx, |project, _| {
            project.structured_execution_store().clone()
        });
        store.update(cx, |store, _| {
            let generation = store.state().project_generation();
            store
                .state_mut()
                .apply_discovery(generation, provider, None)
                .expect("discovery");
        });
        let (container, cx) = cx.add_window_view(|window, cx| {
            workspace::MultiWorkspace::test_new(project.clone(), window, cx)
        });
        let workspace = container.read_with(cx, |container, _| container.workspace().clone());
        let async_cx = cx.update(|window, cx| window.to_async(cx));
        let panel = TestsPanel::load(workspace.downgrade(), async_cx)
            .await
            .expect("Tests panel");
        workspace.update_in(cx, |workspace, window, cx| {
            workspace.add_panel(panel.clone(), window, cx);
            workspace.focus_panel::<TestsPanel>(window, cx);
            window.activate_a11y_for_test();
        });
        cx.run_until_parked();
        panel.update(cx, |panel, cx| {
            panel.active = true;
            panel.refresh(cx);
        });
        cx.run_until_parked();
        let assert_disabled_controls =
            |cx: &mut gpui::VisualTestContext, expected: &[(&str, bool)]| {
                for _ in 0..2 {
                    cx.update(|window, cx| {
                        window.draw(cx).clear(cx);
                        assert!(panel.read(cx).focus_handle.is_focused(window));
                        let data = window.debug_a11y_tree_json().expect("active Tests tree");
                        let data: serde_json::Value =
                            serde_json::from_str(&data).expect("valid Tests tree");
                        let nodes = data["nodes"].as_object().expect("Tests nodes");
                        for (label, disabled) in expected {
                            let control = nodes
                                .values()
                                .find(|node| {
                                    node["aria"]["role"] == "Button"
                                        && node["aria"]["label"] == *label
                                })
                                .expect("Tests toolbar control");
                            assert_eq!(control["aria"]["disabled"], *disabled, "{control}");
                        }
                    });
                }
            };
        assert_eq!(
            panel.read_with(cx, |panel, _| panel.projection.failed.len()),
            1
        );
        assert_disabled_controls(cx, &[("Rerun Failed Tests", false)]);
        let root_id = tree_id(&provider_id, &StructuredNodeId("web-root".into()));
        let mut previous_status_id = None;
        for (index, (phase, diagnostic, label)) in [
            (StructuredRunPhase::Queued, None, "task queued"),
            (StructuredRunPhase::Running, None, "task running"),
            (StructuredRunPhase::Completed, None, "task completed"),
            (
                StructuredRunPhase::Completed,
                Some("Task exited with status 7"),
                "task completed; Task exited with status 7",
            ),
            (
                StructuredRunPhase::SpawnError,
                Some("Task could not start"),
                "task failed to start; Task could not start",
            ),
            (
                StructuredRunPhase::Cancelled,
                Some("Termination is not confirmed"),
                "task cancelled; Termination is not confirmed",
            ),
        ]
        .into_iter()
        .enumerate()
        {
            let run_id = StructuredRunId(format!("panel-aggregate-{index}"));
            store.update(cx, |store, cx| {
                let state = store.state_mut();
                let generation = state.project_generation();
                state
                    .begin_run(
                        generation,
                        &provider_id,
                        StructuredRun::new(
                            run_id.clone(),
                            discovery_generation,
                            vec![StructuredNodeId("browser-suite".into())],
                        ),
                    )
                    .expect("aggregate run");
                state
                    .set_run_phase(generation, &provider_id, &run_id, phase, diagnostic)
                    .expect("aggregate phase");
                cx.emit(StructuredExecutionStoreEvent::Changed(provider_id.clone()));
            });
            cx.run_until_parked();
            panel.update(cx, |panel, cx| {
                let root =
                    find_node(&panel.projection.snapshot.roots, "Web Tests").expect("provider row");
                let expected = format!("{label}; per-test results unavailable");
                assert_eq!(root.secondary_label.as_deref(), Some(expected.as_str()));
                assert_eq!(root.accessibility_label, format!("Web Tests, {expected}"));
                assert_eq!(
                    panel.projection.summary,
                    StructuredExecutionSummary::default()
                );
                assert!(panel.projection.failed.is_empty());
                assert!(
                    panel
                        .projection
                        .summary_label()
                        .starts_with("Per-test results unavailable")
                );
                let child = find_node(&panel.projection.snapshot.roots, "renders login")
                    .expect("unknown child");
                assert!(child.secondary_label.is_none());
                assert_eq!(child.icon, Some(IconName::Circle));
                panel.host.expand_all();
                panel.host.select(root_id.clone());
                panel.host.select_next();
                assert_eq!(
                    panel.selected().expect("keyboard selection").node_id.0,
                    "browser-suite"
                );
                assert_eq!(
                    panel
                        .selected()
                        .expect("current action selection")
                        .run_id
                        .as_ref(),
                    Some(&run_id)
                );
                panel.host.select_next();
                assert_eq!(
                    panel.selected().expect("unknown child selection").node_id.0,
                    "renders-home"
                );
                cx.notify();
            });
            cx.update(|window, cx| window.draw(cx).clear(cx));
            panel.read_with(cx, |panel, _| {
                let fixture_ids = ["web-root", "browser-suite", "renders-home", "renders-login"]
                    .map(|id| tree_id(&provider_id, &StructuredNodeId(id.into())))
                    .into_iter()
                    .collect::<HashSet<_>>();
                let fixture_rows = panel
                    .host
                    .visible_rows()
                    .iter()
                    .filter(|row| fixture_ids.contains(&row.id))
                    .count();
                assert_eq!(fixture_rows, fixture_ids.len());
                let other_roots = panel
                    .projection
                    .snapshot
                    .roots
                    .iter()
                    .filter(|root| root.id != root_id)
                    .collect::<Vec<_>>();
                assert!(other_roots.iter().all(|root| root.children.is_empty()));
                assert_eq!(
                    panel.host.visible_rows().len(),
                    fixture_rows + other_roots.len()
                );
                assert!(
                    panel
                        .host
                        .scroll_handle()
                        .0
                        .borrow()
                        .last_item_size
                        .is_some()
                );
            });
            let summary = panel.read_with(cx, |panel, _| panel.projection.summary_label());
            for _ in 0..2 {
                cx.update(|window, cx| {
                    window.draw(cx).clear(cx);
                    assert!(panel.read(cx).focus_handle.is_focused(window));
                    let data = window.debug_a11y_tree_json().expect("active Tests tree");
                    let data: serde_json::Value =
                        serde_json::from_str(&data).expect("valid Tests tree");
                    let status = data["nodes"]
                        .as_object()
                        .expect("Tests nodes")
                        .values()
                        .find(|node| node["aria"]["label"] == "Test results")
                        .expect("Tests summary status");
                    assert_eq!(status["aria"]["role"], "Status");
                    assert_eq!(status["aria"]["value"], summary);
                    assert_eq!(status["aria"]["live"], "Polite");
                    if let Some(previous) = &previous_status_id {
                        assert_eq!(&status["accesskit_id"], previous);
                    }
                    previous_status_id = Some(status["accesskit_id"].clone());
                });
            }
            assert_disabled_controls(
                cx,
                &[
                    ("Run Selected Tests", false),
                    ("Debug Selected Test", false),
                    ("Cancel Test Run", false),
                    ("Rerun Failed Tests", true),
                    ("Show Task Terminal", false),
                ],
            );
            eprintln!(
                "aggregate-tests-panel-painted phase={phase:?} child_outcomes=unknown summary=unavailable"
            );
        }
        let mut previous_action_status_id = None;
        for (action, expected) in [
            (
                Box::new(DebugSelectedTests) as Box<dyn gpui::Action>,
                if cfg!(all(
                    feature = "rust-test-actions",
                    feature = "jvm-test-actions"
                )) {
                    "The selected structured test provider is unavailable"
                } else if cfg!(feature = "rust-test-actions") {
                    "The selected structured provider is not a Rust test provider"
                } else if cfg!(feature = "jvm-test-actions") {
                    "The selected structured provider is not a Java test provider"
                } else {
                    "No structured test provider can debug this selection"
                },
            ),
            (
                Box::new(RerunFailedTests) as Box<dyn gpui::Action>,
                "No failed tests are available to rerun",
            ),
        ] {
            cx.update(|window, cx| window.dispatch_action(action, cx));
            cx.run_until_parked();
            for _ in 0..2 {
                cx.update(|window, cx| {
                    window.draw(cx).clear(cx);
                    assert!(panel.read(cx).focus_handle.is_focused(window));
                    let data = window.debug_a11y_tree_json().expect("active Tests tree");
                    let data: serde_json::Value =
                        serde_json::from_str(&data).expect("valid Tests tree");
                    let status = data["nodes"]
                        .as_object()
                        .expect("Tests nodes")
                        .values()
                        .find(|node| node["aria"]["label"] == "Test action status")
                        .expect("Tests action status");
                    assert_eq!(status["aria"]["role"], "Status");
                    assert_eq!(status["aria"]["value"], expected);
                    assert_eq!(status["aria"]["live"], "Polite");
                    if let Some(previous) = &previous_action_status_id {
                        assert_eq!(&status["accesskit_id"], previous);
                    }
                    previous_action_status_id = Some(status["accesskit_id"].clone());
                });
            }
        }
        cx.deactivate_window();
        container.update_in(cx, |_, window, _| window.remove_window());
    }

    #[test]
    fn test_explorer_projects_non_rust_results_and_filters_stably() {
        let provider = fake_provider(StructuredProviderStatus::Current);
        let all = project_test_explorer(
            std::slice::from_ref(&provider),
            &TestExplorerFilter::default(),
        );
        let again = project_test_explorer(
            std::slice::from_ref(&provider),
            &TestExplorerFilter::default(),
        );
        assert_eq!(all.snapshot.roots[0].id, again.snapshot.roots[0].id);
        assert_eq!(all.summary.total, 2);
        assert_eq!(all.failed.len(), 1);
        assert!(
            all.snapshot.roots[0]
                .accessibility_label
                .contains("Web Tests")
        );
        assert!(
            all.navigation
                .values()
                .any(|value| value == &path("web/login.test.js"))
        );
        let provider_node =
            find_node(&all.snapshot.roots, "Web Tests").expect("provider root should be projected");
        let suite_node =
            find_node(&all.snapshot.roots, "Browser").expect("suite should be projected");
        let case_node =
            find_node(&all.snapshot.roots, "renders login").expect("case should be projected");
        assert!(!all.can_run(&provider_node.id));
        assert!(!all.can_debug(&provider_node.id));
        assert!(all.can_run(&suite_node.id));
        assert!(!all.can_debug(&suite_node.id));
        assert!(all.can_run(&case_node.id));
        assert!(all.can_debug(&case_node.id));

        let failed = project_test_explorer(
            std::slice::from_ref(&provider),
            &TestExplorerFilter {
                query: String::new(),
                status: TestStatusFilter::Failed,
            },
        );
        assert!(find_node(&failed.snapshot.roots, "renders login").is_some());
        assert!(find_node(&failed.snapshot.roots, "renders home").is_none());

        let query = project_test_explorer(
            &[provider],
            &TestExplorerFilter {
                query: "login".to_string(),
                status: TestStatusFilter::All,
            },
        );
        assert!(find_node(&query.snapshot.roots, "renders login").is_some());
        assert!(find_node(&query.snapshot.roots, "renders home").is_none());
        assert!(
            collect_ids(&query.snapshot.roots)
                .iter()
                .all(|id| !id.0.contains("Cargo") && !id.0.contains("Rust"))
        );
    }

    #[cfg(feature = "jvm-test-actions")]
    #[test]
    fn java_test_projection_only_runs_cases() {
        let mut provider = fake_provider(StructuredProviderStatus::Current);
        provider.provider_id = StructuredProviderId(JVM_JUNIT_PROVIDER_ID.to_string());
        let projection = project_test_explorer(&[provider], &TestExplorerFilter::default());
        let suite = find_node(&projection.snapshot.roots, "Browser").expect("Java suite");
        let case = find_node(&projection.snapshot.roots, "renders login").expect("Java case");
        assert!(!projection.can_run(&suite.id));
        assert!(projection.can_run(&case.id));
    }

    #[test]
    fn test_explorer_maps_every_provider_state_distinctly() {
        let cases = [
            (StructuredProviderStatus::Loading, "loading"),
            (StructuredProviderStatus::Empty, "No tests"),
            (StructuredProviderStatus::Partial, "Some tests"),
            (StructuredProviderStatus::Stale, "stale"),
            (StructuredProviderStatus::Error, "failed"),
            (StructuredProviderStatus::Restricted, "Trust"),
            (StructuredProviderStatus::Disconnected, "Reconnect"),
            (StructuredProviderStatus::Mismatch, "protocol"),
        ];
        for (status, expected) in cases {
            let projection =
                project_test_explorer(&[fake_provider(status)], &TestExplorerFilter::default());
            let message = match projection.snapshot.status {
                LanguageToolProviderStatus::Loading => "loading".to_string(),
                LanguageToolProviderStatus::Empty(message)
                | LanguageToolProviderStatus::Partial(message)
                | LanguageToolProviderStatus::Stale(message)
                | LanguageToolProviderStatus::Restricted(message)
                | LanguageToolProviderStatus::Unsupported(message)
                | LanguageToolProviderStatus::Mismatch(message)
                | LanguageToolProviderStatus::Error(message)
                | LanguageToolProviderStatus::Disconnected(message) => message,
                LanguageToolProviderStatus::Current => "current".to_string(),
            };
            assert!(
                message.to_lowercase().contains(&expected.to_lowercase()),
                "{status:?} produced {message:?}"
            );
        }
    }

    #[test]
    fn test_explorer_persistence_contains_only_filters_and_opaque_ids() {
        let state = PersistedTestsPanelState {
            version: TESTS_PANEL_STATE_VERSION,
            query: "login".to_string(),
            status: TestStatusFilter::Failed,
            selected_provider: Some("web-tests".to_string()),
            selected_node: Some("renders-login".to_string()),
        };
        let serialized = serde_json::to_string(&state).expect("state should serialize");
        assert!(!serialized.contains("expected dashboard"));
        assert!(!serialized.contains("/Users/"));
        assert!(!serialized.contains("project_env"));
        assert!(!serialized.contains("events"));
        assert_eq!(
            <TestsPanel as Panel>::persistent_name(),
            "Tests",
            "the panel title is a resolved product decision"
        );
    }

    #[test]
    fn rust_workspace_test_explorer_state_migration_is_bounded_and_private() {
        let state = PersistedTestsPanelState {
            version: 0,
            query: "q".repeat(MAX_FILTER_BYTES + 32),
            status: TestStatusFilter::Failed,
            selected_provider: Some("provider".repeat(MAX_FILTER_BYTES)),
            selected_node: Some("node".repeat(MAX_FILTER_BYTES)),
        }
        .sanitized();
        let serialized = serde_json::to_string(&state).expect("state should serialize");

        assert_eq!(state.version, TESTS_PANEL_STATE_VERSION);
        assert_eq!(state.query.len(), MAX_FILTER_BYTES);
        assert!(
            state
                .selected_provider
                .is_some_and(|value| value.len() <= MAX_FILTER_BYTES)
        );
        assert!(
            state
                .selected_node
                .is_some_and(|value| value.len() <= MAX_FILTER_BYTES)
        );
        assert!(!serialized.contains("/Users/"));
        assert!(!serialized.contains("environment"));
        assert!(!serialized.contains("terminal"));
    }

    #[test]
    fn rust_test_actions_expose_stable_delegate_selections() {
        let provider = fake_provider(StructuredProviderStatus::Current);
        let projection = project_test_explorer(&[provider], &TestExplorerFilter::default());
        let failed_node = projection
            .snapshot
            .roots
            .iter()
            .find_map(|node| find_node(std::slice::from_ref(node), "renders login"))
            .expect("failed case should be visible");
        let selection = projection
            .selection(&failed_node.id)
            .expect("an actionable node should expose an opaque provider selection");

        assert_eq!(selection.provider_id.0, "web-tests");
        assert_eq!(selection.discovery_generation, DiscoveryGeneration(3));
        assert_eq!(selection.node_id.0, "renders-login");
        assert_eq!(
            selection.run_id.as_ref().map(|run| run.0.as_str()),
            Some("run-1")
        );
        assert_eq!(projection.failed(), std::slice::from_ref(selection));
    }

    #[gpui::test]
    #[cfg(feature = "rust-test-actions")]
    async fn test_explorer_opens_real_rust_lsp_source_from_case(cx: &mut TestAppContext) {
        use futures::StreamExt as _;
        use language::{FakeLspAdapter, rust_lang};
        use project::lsp_store::lsp_ext_command::{
            CargoRunnableArgs, Runnable, RunnableArgs, Runnables,
        };
        use std::{
            path::Path,
            sync::atomic::{AtomicUsize, Ordering},
        };

        crate::tests::init_test(cx);
        cx.executor().allow_parking();
        let root = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../project/test_data/rust_test_provider/physical-workspace")
            .canonicalize()
            .expect("physical Rust fixture");
        let source = root.join("tests/integration.rs");
        let project = Project::test(
            Arc::new(fs::RealFs::new(None, cx.executor())),
            [root.as_path()],
            cx,
        )
        .await;
        let languages = project.read_with(cx, |project, _| project.languages().clone());
        languages.add(rust_lang());
        let mut fake_servers = languages.register_fake_lsp(
            "Rust",
            FakeLspAdapter {
                name: "rust-analyzer",
                ..FakeLspAdapter::default()
            },
        );
        let (_buffer, _handle) = project
            .update(cx, |project, cx| {
                project.open_local_buffer_with_lsp(source.as_path(), cx)
            })
            .await
            .expect("open Rust source with LSP");
        let fake_server = fake_servers.next().await.expect("fake Rust Analyzer");
        let source_uri = lsp::Uri::from_file_path(&source).expect("source URI");
        let root_for_runnable = root.clone();
        let request_count = Arc::new(AtomicUsize::new(0));
        fake_server.set_request_handler::<Runnables, _, _>({
            let request_count = request_count.clone();
            move |_, _| {
                request_count.fetch_add(1, Ordering::SeqCst);
                let source_uri = source_uri.clone();
                let root = root_for_runnable.clone();
                async move {
                    Ok(vec![Runnable {
                        label: "test integration_passes".into(),
                        location: Some(lsp::LocationLink {
                            origin_selection_range: None,
                            target_uri: source_uri,
                            target_range: lsp::Range::new(
                                lsp::Position::new(1, 3),
                                lsp::Position::new(1, 21),
                            ),
                            target_selection_range: lsp::Range::new(
                                lsp::Position::new(1, 3),
                                lsp::Position::new(1, 21),
                            ),
                        }),
                        args: RunnableArgs::Cargo(CargoRunnableArgs {
                            environment: HashMap::default(),
                            cwd: root,
                            override_cargo: None,
                            workspace_root: None,
                            cargo_args: vec![
                                "test".into(),
                                "--package".into(),
                                "zed-rust-tools-physical-fixture".into(),
                                "--test".into(),
                                "integration".into(),
                            ],
                            executable_args: vec!["integration_passes".into(), "--exact".into()],
                        }),
                    }])
                }
            }
        });
        cx.run_until_parked();
        let provider =
            project.read_with(cx, |project, _| project.rust_test_provider_store().clone());
        let mut attempt = 0;
        let snapshot = loop {
            attempt += 1;
            match provider.update(cx, |store, cx| store.refresh(cx)).await {
                Ok(snapshot) => break snapshot,
                Err(error)
                    if attempt < 3
                        && error
                            .to_string()
                            .contains("stale Rust test discovery generation") =>
                {
                    cx.run_until_parked();
                }
                Err(error) => panic!("Rust discovery with LSP source: {error:#}"),
            }
        };
        let request_count_after_discovery = request_count.load(Ordering::SeqCst);
        assert!(request_count_after_discovery > 0);
        let case = snapshot
            .nodes
            .iter()
            .find(|node| node.label == "integration_passes")
            .expect("integration test case");
        let source_path = case.path.clone().expect("LSP source path");

        let (container, cx) = cx.add_window_view(|window, cx| {
            workspace::MultiWorkspace::test_new(project.clone(), window, cx)
        });
        let workspace = container.read_with(cx, |container, _| container.workspace().clone());
        let async_cx = cx.update(|window, cx| window.to_async(cx));
        let panel = TestsPanel::load(workspace.downgrade(), async_cx)
            .await
            .expect("Tests panel");
        panel.update(cx, |panel, cx| {
            panel.active = true;
            panel.refresh(cx);
        });
        cx.run_until_parked();
        assert_eq!(
            request_count.load(Ordering::SeqCst),
            request_count_after_discovery,
            "passive panel projection must not request runnables"
        );
        panel.update_in(cx, |panel, window, cx| {
            let id = tree_id(&snapshot.provider_id, &case.id);
            assert_eq!(panel.projection.navigation(&id), Some(&source_path));
            panel.host.expand_all();
            panel.host.select(id);
            assert!(panel.host.can_activate());
            panel.activate_selected(window, cx);
        });
        cx.run_until_parked();
        assert_eq!(
            workspace.read_with(cx, |workspace, cx| {
                workspace
                    .active_item(cx)
                    .and_then(|item| item.project_path(cx))
            }),
            Some(source_path.clone())
        );
        let other_source = project.read_with(cx, |project, cx| {
            project
                .find_project_path(root.join("src/lib.rs"), cx)
                .expect("other fixture source")
        });
        workspace
            .update_in(cx, |workspace, window, cx| {
                workspace.open_path(other_source.clone(), None, true, window, cx)
            })
            .await
            .expect("open other fixture source");
        cx.update(|_, cx| {
            cx.update_global::<settings::SettingsStore, _>(|store, cx| {
                store.update_user_settings(cx, |settings| {
                    settings.project.worktree.private_files =
                        Some(settings::ExtendingVec(vec!["**/integration.rs".into()]));
                });
            });
        });
        cx.run_until_parked();
        project.read_with(cx, |project, cx| {
            assert!(
                project
                    .entry_for_path(&source_path, cx)
                    .expect("original test source entry")
                    .is_private
            );
        });
        panel.update_in(cx, |panel, window, cx| {
            let id = tree_id(&snapshot.provider_id, &case.id);
            panel.host.select(id.clone());
            assert!(panel.current_source_path(&id, cx).is_none());
            panel.activate_selected(window, cx);
        });
        cx.run_until_parked();
        assert_eq!(
            workspace.read_with(cx, |workspace, cx| {
                workspace
                    .active_item(cx)
                    .and_then(|item| item.project_path(cx))
            }),
            Some(other_source),
            "a former Tests source link must not reopen a newly private file"
        );
        cx.deactivate_window();
        container.update_in(cx, |_, window, _| window.remove_window());
    }

    #[gpui::test]
    #[cfg(feature = "rust-test-actions")]
    async fn rust_doctest_dispatch_retains_original_opt_in_while_task_contexts_are_held(
        cx: &mut TestAppContext,
    ) {
        crate::tests::init_test(cx);
        let project = Project::test(fs::FakeFs::new(cx.executor()), [], cx).await;
        let (_, cx) = cx
            .add_window_view(|window, cx| workspace::MultiWorkspace::test_new(project, window, cx));
        let feature = "rust-exact-doctest-actions-experimental";
        for (experimental, revoke, succeeds) in [
            (true, true, false),
            (true, false, true),
            (false, true, true),
        ] {
            cx.update(|_, cx| {
                project::task_store::set_experimental_feature_for_test(feature, true, cx)
            });
            let requirements = if experimental {
                vec![feature.to_string()]
            } else {
                Vec::new()
            };
            let pending_operation = cx
                .update(|_, cx| PendingExperimentalFeatureOperation::new(&requirements, cx))
                .expect("original selection opt-in");
            let plan = RustTestActionPlan::Task {
                run_id: StructuredRunId("context-held-case".into()),
                discovery_generation: DiscoveryGeneration(1),
                scope_node_ids: vec![StructuredNodeId("case".into())],
                worktree_id: WorktreeId::from_usize(1),
                template: task::TaskTemplate {
                    required_experimental_features: requirements,
                    experimental_policy_version: task::EXPERIMENTAL_TASK_POLICY_VERSION,
                    label: "context-held case".into(),
                    command: "cargo".into(),
                    args: vec!["test".into()],
                    ..Default::default()
                },
            };
            let (release, held_contexts) = futures::channel::oneshot::channel();
            let task_contexts = cx.background_spawn(async move {
                held_contexts.await.expect("held Task contexts released");
                project::TaskContexts::default()
            });
            let continuation = cx.update(|window, cx| {
                window.spawn(cx, async move |mut cx| {
                    await_rust_test_dispatch(
                        gpui::Task::ready(Ok(plan)),
                        task_contexts,
                        &pending_operation,
                        &mut cx,
                    )
                    .await
                })
            });
            cx.run_until_parked();
            if revoke {
                cx.update(|_, cx| {
                    project::task_store::set_experimental_feature_for_test(feature, false, cx)
                });
                cx.run_until_parked();
                cx.update(|_, cx| {
                    project::task_store::set_experimental_feature_for_test(feature, true, cx)
                });
                cx.run_until_parked();
            }
            release
                .send(())
                .expect("original Task context receiver retained");
            assert_eq!(
                continuation.await.is_ok(),
                succeeds,
                "only a fresh opted-in action or ordinary action may continue"
            );
        }
        cx.update(|window, _| window.remove_window());
    }

    #[gpui::test]
    #[cfg(feature = "rust-test-actions")]
    async fn rust_test_rerun_failed_skips_stale_and_runs_current_cases(cx: &mut TestAppContext) {
        use collections::HashSet;
        use project::{
            rust_test_provider::RustTestProviderStore,
            trusted_worktrees::{PathTrust, TrustedWorktrees},
        };
        use std::path::Path;

        crate::tests::init_test(cx);
        cx.executor().allow_parking();
        cx.update(|cx| terminal_view::init(cx));
        let root = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../project/test_data/rust_test_provider/physical-workspace")
            .canonicalize()
            .expect("physical Rust fixture");
        let project = Project::test(
            Arc::new(fs::RealFs::new(None, cx.executor())),
            [root.as_path()],
            cx,
        )
        .await;
        let (worktree_store, worktree_id) = project.read_with(cx, |project, cx| {
            let worktree_store = project.worktree_store();
            let worktree_id = worktree_store
                .read(cx)
                .worktrees()
                .next()
                .expect("Rust fixture worktree")
                .read(cx)
                .id();
            (worktree_store, worktree_id)
        });
        cx.update(|cx| project::trusted_worktrees::init(collections::HashMap::default(), cx));
        let trust = cx.read(|cx| TrustedWorktrees::try_get_global(cx).expect("trust initialized"));
        trust.update(cx, |trust, cx| {
            trust.trust(
                &worktree_store,
                HashSet::from_iter([PathTrust::Worktree(worktree_id)]),
                cx,
            )
        });
        let (container, cx) = cx.add_window_view(|window, cx| {
            workspace::MultiWorkspace::test_new(project.clone(), window, cx)
        });
        let workspace = container.read_with(cx, |container, _| container.workspace().clone());
        let async_cx = cx.update(|window, cx| window.to_async(cx));
        let terminal_panel =
            terminal_view::terminal_panel::TerminalPanel::load(workspace.downgrade(), async_cx)
                .await
                .expect("production terminal panel");
        workspace.update_in(cx, |workspace, window, cx| {
            workspace.add_panel(terminal_panel.clone(), window, cx);
        });
        cx.run_until_parked();
        cx.background_executor
            .timer(Duration::from_millis(300))
            .await;
        cx.run_until_parked();
        let provider =
            project.read_with(cx, |project, _| project.rust_test_provider_store().clone());
        let snapshot = provider
            .update(cx, |provider, cx| provider.refresh(cx))
            .await
            .expect("discover real Cargo cases");
        let cases = ["unit_passes", "integration_passes"].map(|label| {
            snapshot
                .nodes
                .iter()
                .find(|node| node.label == label)
                .expect("fixture case")
                .id
                .clone()
        });
        let provider_id = RustTestProviderStore::provider_id();
        let failed_run_id = StructuredRunId("rust-rerun-failed-fixture".to_string());
        let structured = project.read_with(cx, |project, _| {
            project.structured_execution_store().clone()
        });
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
                        cases.to_vec(),
                    ),
                )
                .expect("failed fixture run");
            for (sequence, node_id) in cases.iter().enumerate() {
                store
                    .state_mut()
                    .apply_event(
                        project_generation,
                        &provider_id,
                        &failed_run_id,
                        StructuredExecutionEvent {
                            sequence: sequence as u64,
                            node_id: node_id.clone(),
                            state: StructuredNodeState::Failed,
                            duration_millis: None,
                            message: Some("fixture failure".to_string()),
                            location: None,
                        },
                        None,
                    )
                    .expect("failed fixture case");
            }
            store
                .state_mut()
                .set_run_phase(
                    project_generation,
                    &provider_id,
                    &failed_run_id,
                    project::structured_execution::StructuredRunPhase::Completed,
                    None,
                )
                .expect("completed failed fixture run");
            cx.emit(StructuredExecutionStoreEvent::Changed(provider_id.clone()));
        });
        let current = structured
            .read_with(cx, |store, _| store.state().provider(&provider_id).cloned())
            .expect("current Rust provider");
        let projection = project_test_explorer(&[current], &TestExplorerFilter::default());
        let selections = cases.map(|node_id| {
            projection
                .failed()
                .iter()
                .find(|selection| selection.node_id == node_id)
                .expect("failed current case")
                .clone()
        });
        let stale = selections[0].clone();
        let snapshot = provider
            .update(cx, |provider, cx| provider.refresh(cx))
            .await
            .expect("supersede failed Rust discovery");
        assert_ne!(snapshot.discovery_generation, stale.discovery_generation);
        let cases = ["unit_passes", "integration_passes"].map(|label| {
            snapshot
                .nodes
                .iter()
                .find(|node| node.label == label)
                .expect("current fixture case")
                .id
                .clone()
        });
        let current_failed_run_id = StructuredRunId("rust-rerun-current-failure".to_string());
        structured.update(cx, |store, cx| {
            let project_generation = store.state().project_generation();
            store
                .state_mut()
                .begin_run(
                    project_generation,
                    &provider_id,
                    StructuredRun::new(
                        current_failed_run_id.clone(),
                        snapshot.discovery_generation,
                        cases.to_vec(),
                    ),
                )
                .expect("current failed fixture run");
            for (sequence, node_id) in cases.iter().enumerate() {
                store
                    .state_mut()
                    .apply_event(
                        project_generation,
                        &provider_id,
                        &current_failed_run_id,
                        StructuredExecutionEvent {
                            sequence: sequence as u64,
                            node_id: node_id.clone(),
                            state: StructuredNodeState::Failed,
                            duration_millis: None,
                            message: Some("current fixture failure".to_string()),
                            location: None,
                        },
                        None,
                    )
                    .expect("current failed fixture case");
            }
            store
                .state_mut()
                .set_run_phase(
                    project_generation,
                    &provider_id,
                    &current_failed_run_id,
                    project::structured_execution::StructuredRunPhase::Completed,
                    None,
                )
                .expect("completed current failed fixture run");
            cx.emit(StructuredExecutionStoreEvent::Changed(provider_id.clone()));
        });
        let current = structured
            .read_with(cx, |store, _| store.state().provider(&provider_id).cloned())
            .expect("current Rust provider");
        let projection = project_test_explorer(&[current], &TestExplorerFilter::default());
        let selections = cases.map(|node_id| {
            projection
                .failed()
                .iter()
                .find(|selection| selection.node_id == node_id)
                .expect("current failed case")
                .clone()
        });
        assert_eq!(
            provider.read_with(cx, |provider, _| provider.generation()),
            snapshot.discovery_generation,
            "the discovery generation must remain current before rerun"
        );
        assert_eq!(
            rust_rerun_summary(2, 1),
            "Ran 2 failed Rust tests; skipped 1 unavailable, removed, stale, or over-limit selections"
        );
        let initial_notifications =
            workspace.read_with(cx, |workspace, _| workspace.notification_ids().len());
        assert!(!cx.read(|cx| rust_failed_selection_is_current(&stale, &project, cx)));
        cx.update(|window, cx| {
            RustTestExplorerDelegate
                .rerun_failed(
                    &[stale, selections[0].clone(), selections[1].clone()],
                    &workspace.downgrade(),
                    window,
                    cx,
                )
                .expect("rerun current Rust cases")
        });
        let mut completed = Vec::new();
        for _ in 0..100 {
            cx.run_until_parked();
            completed = structured.read_with(cx, |store, _| {
                let provider = store.state().provider(&provider_id).expect("Rust provider");
                let mut seen = HashSet::default();
                provider
                    .completed_runs
                    .iter()
                    .chain(provider.last_complete_run.iter())
                    .filter(|run| {
                        run.id != failed_run_id
                            && run.id != current_failed_run_id
                            && seen.insert(run.id.clone())
                    })
                    .cloned()
                    .collect::<Vec<_>>()
            });
            if completed.len() >= 2 {
                break;
            }
            cx.background_executor
                .timer(Duration::from_millis(100))
                .await;
        }
        assert_eq!(
            completed.len(),
            2,
            "both current cases must run to completion; provider: {:#?}",
            structured.read_with(cx, |store, _| store.state().provider(&provider_id).cloned())
        );
        assert_eq!(completed[0].scope_node_ids.len(), 1);
        assert_eq!(completed[1].scope_node_ids.len(), 1);
        assert_ne!(completed[0].scope_node_ids, completed[1].scope_node_ids);
        assert!(
            completed.iter().all(|run| {
                run.phase == project::structured_execution::StructuredRunPhase::Completed
            }),
            "rerun phases: {completed:#?}"
        );
        cx.run_until_parked();
        assert!(
            workspace.read_with(cx, |workspace, _| workspace.notification_ids().len())
                > initial_notifications,
            "the skipped selection summary must be visible in the workspace"
        );
        container.update_in(cx, |_, window, _| window.remove_window());
    }

    #[gpui::test]
    #[cfg(feature = "rust-test-actions")]
    async fn rust_test_rerun_failed_latches_doctest_revocation_behind_actual_ordinary_child(
        cx: &mut TestAppContext,
    ) {
        use project::{
            rust_test_provider::RustTestProviderStore,
            trusted_worktrees::{PathTrust, TrustedWorktrees},
        };

        crate::tests::init_test(cx);
        cx.executor().allow_parking();
        cx.update(|cx| {
            terminal_view::init(cx);
            project::task_store::set_experimental_feature_for_test(
                "rust-exact-doctest-actions-experimental",
                true,
                cx,
            );
        });
        let owned = tempfile::TempDir::new().expect("owned Cargo workload fixture");
        let root = owned.path().join("workspace");
        std::fs::create_dir_all(root.join("src")).expect("fixture source directory");
        let root = root
            .canonicalize()
            .expect("canonical owned Cargo workspace");
        let started = owned.path().join("ordinary-started");
        let ordinary_finished = owned.path().join("ordinary-finished");
        let exact_finished = owned.path().join("exact-finished");
        std::fs::write(root.join("Cargo.toml"),
            "[package]\nname = \"doctest-revocation-fixture\"\nversion = \"0.1.0\"\nedition = \"2021\"\n[workspace]\n"
        ).expect("fixture manifest");
        std::fs::write(root.join("Cargo.lock"),
            "# This file is automatically @generated by Cargo.\n# It is not intended for manual editing.\nversion = 4\n\n[[package]]\nname = \"doctest-revocation-fixture\"\nversion = \"0.1.0\"\n"
        ).expect("canonical fixture lockfile");
        std::fs::write(root.join("src/lib.rs"), format!(r#"
/// ```
/// std::fs::write({exact_finished:?}, b"exact").expect("owned Exact marker");
/// ```
pub fn example() {{}}
#[cfg(test)]
mod tests {{
    #[test]
    fn ordinary_held() {{
        std::fs::write({started:?}, b"started").expect("owned start marker");
        let mut byte = [0];
        std::io::Read::read_exact(&mut std::io::stdin(), &mut byte).expect("release original child");
        std::fs::write({ordinary_finished:?}, b"finished").expect("owned completion marker");
    }}
}}
"#)).expect("fixture real Cargo workloads");
        let project = Project::test(
            Arc::new(fs::RealFs::new(None, cx.executor())),
            [root.as_path()],
            cx,
        )
        .await;
        let (worktree_store, worktree_id) = project.read_with(cx, |project, cx| {
            let worktree_store = project.worktree_store();
            let worktree_id = worktree_store
                .read(cx)
                .worktrees()
                .next()
                .expect("owned Rust worktree")
                .read(cx)
                .id();
            (worktree_store, worktree_id)
        });
        cx.update(|cx| project::trusted_worktrees::init(collections::HashMap::default(), cx));
        let trust = cx.read(|cx| TrustedWorktrees::try_get_global(cx).expect("trust initialized"));
        trust.update(cx, |trust, cx| {
            trust.trust(
                &worktree_store,
                collections::HashSet::from_iter([PathTrust::Worktree(worktree_id)]),
                cx,
            )
        });
        let (container, cx) = cx.add_window_view(|window, cx| {
            workspace::MultiWorkspace::test_new(project.clone(), window, cx)
        });
        let workspace = container.read_with(cx, |container, _| container.workspace().clone());
        let async_cx = cx.update(|window, cx| window.to_async(cx));
        let terminal_panel =
            terminal_view::terminal_panel::TerminalPanel::load(workspace.downgrade(), async_cx)
                .await
                .expect("production TerminalPanel");
        workspace.update_in(cx, |workspace, window, cx| {
            workspace.add_panel(terminal_panel.clone(), window, cx);
        });
        cx.run_until_parked();
        cx.background_executor
            .timer(Duration::from_millis(300))
            .await;
        cx.run_until_parked();
        let provider =
            project.read_with(cx, |project, _| project.rust_test_provider_store().clone());
        let snapshot = provider
            .update(cx, |provider, cx| provider.refresh(cx))
            .await
            .expect("actual installed Cargo discovery");
        let ordinary = snapshot
            .nodes
            .iter()
            .find(|node| node.kind == StructuredNodeKind::Case && node.label == "ordinary_held")
            .expect("actual ordinary harness case")
            .id
            .clone();
        let exact = snapshot
            .nodes
            .iter()
            .find(|node| {
                node.kind == StructuredNodeKind::Case
                    && node.experimental_feature.as_deref()
                        == Some("rust-exact-doctest-actions-experimental")
            })
            .expect("actual rustdoc case");
        assert_eq!(exact.experimental_feature_enabled, Some(true));
        let exact = exact.id.clone();
        assert!(
            !started.exists() && !ordinary_finished.exists() && !exact_finished.exists(),
            "discovery must list without executing either selected workload"
        );
        let cases = [ordinary.clone(), exact.clone()];
        let provider_id = RustTestProviderStore::provider_id();
        let failed_run_id = StructuredRunId("held-ordinary-before-exact-failed-selection".into());
        let structured = project.read_with(cx, |project, _| {
            project.structured_execution_store().clone()
        });
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
                        cases.to_vec(),
                    ),
                )
                .expect("prior failed selection DATA");
            for (sequence, node) in cases.iter().enumerate() {
                store
                    .state_mut()
                    .apply_event(
                        project_generation,
                        &provider_id,
                        &failed_run_id,
                        StructuredExecutionEvent {
                            sequence: sequence as u64,
                            node_id: node.clone(),
                            state: StructuredNodeState::Failed,
                            duration_millis: None,
                            message: Some("prior fixture failure".into()),
                            location: None,
                        },
                        None,
                    )
                    .expect("prior failed selected case DATA");
            }
            store
                .state_mut()
                .set_run_phase(
                    project_generation,
                    &provider_id,
                    &failed_run_id,
                    StructuredRunPhase::Completed,
                    None,
                )
                .expect("prior failed selection completed DATA");
            cx.emit(StructuredExecutionStoreEvent::Changed(provider_id.clone()));
        });
        let current = structured
            .read_with(cx, |store, _| store.state().provider(&provider_id).cloned())
            .expect("current provider");
        let projection = project_test_explorer(&[current], &TestExplorerFilter::default());
        let selections = cases.map(|node| {
            projection
                .failed()
                .iter()
                .find(|selection| selection.node_id == node)
                .expect("failed selected case")
                .clone()
        });
        let initial_notifications =
            workspace.read_with(cx, |workspace, _| workspace.notification_ids());
        cx.update(|window, cx| {
            RustTestExplorerDelegate
                .rerun_failed(&selections, &workspace.downgrade(), window, cx)
                .expect("production rerun-failed batch")
        });
        for _ in 0..100 {
            cx.run_until_parked();
            if started.exists() {
                break;
            }
            cx.background_executor
                .timer(Duration::from_millis(100))
                .await;
        }
        assert!(
            started.exists(),
            "earlier ordinary actual Cargo child must be held"
        );
        assert!(!ordinary_finished.exists() && !exact_finished.exists());
        structured.read_with(cx, |store, _| {
            let run = store
                .state()
                .provider(&provider_id)
                .expect("provider")
                .current_run
                .as_ref()
                .expect("earlier actual run remains active");
            assert_eq!(run.scope_node_ids, vec![ordinary.clone()]);
            assert_eq!(run.phase, StructuredRunPhase::Running);
        });
        for enabled in [false, true] {
            cx.update(|_, cx| {
                project::task_store::set_experimental_feature_for_test(
                    "rust-exact-doctest-actions-experimental",
                    enabled,
                    cx,
                )
            });
            cx.run_until_parked();
        }
        let ordinary_terminal = terminal_panel.read_with(cx, |panel, cx| {
            panel
                .panes()
                .into_iter()
                .find_map(|pane| {
                    pane.read(cx).items().find_map(|item| {
                        let view = item.downcast::<terminal_view::TerminalView>()?;
                        let terminal = view.read(cx).terminal().clone();
                        terminal
                            .read(cx)
                            .task()
                            .is_some_and(|task| {
                                task.spawned_task.full_label == "Rust test tests::ordinary_held"
                            })
                            .then_some(terminal)
                    })
                })
                .expect("production Task PTY owner")
        });
        ordinary_terminal.update(cx, |terminal, _| terminal.input(b"release\n".to_vec()));
        let mut batch_completed = false;
        for _ in 0..100 {
            cx.run_until_parked();
            batch_completed = structured.read_with(cx, |store, _| {
                let provider = store.state().provider(&provider_id).expect("provider");
                provider
                    .current_run
                    .as_ref()
                    .is_some_and(|run| run.phase.is_terminal())
                    && provider
                        .completed_runs
                        .iter()
                        .chain(provider.last_complete_run.iter())
                        .any(|run| {
                            run.id != failed_run_id
                                && run.scope_node_ids == vec![ordinary.clone()]
                                && run.phase == StructuredRunPhase::Completed
                        })
            }) && workspace.read_with(cx, |workspace, _| {
                workspace
                    .notification_ids()
                    .iter()
                    .any(|id| !initial_notifications.contains(id))
            });
            if batch_completed {
                break;
            }
            cx.background_executor
                .timer(Duration::from_millis(100))
                .await;
        }
        assert!(
            batch_completed,
            "ordinary child completes and batch reports omitted Exact selection"
        );
        assert!(ordinary_finished.exists());
        assert!(
            !exact_finished.exists(),
            "restoring opt-in cannot revive later original Exact selection"
        );
        structured.read_with(cx, |store, _| {
            let provider = store.state().provider(&provider_id).expect("provider");
            assert_eq!(provider.discovery_generation, snapshot.discovery_generation);
            let completed = provider
                .completed_runs
                .iter()
                .chain(provider.last_complete_run.iter())
                .find(|run| run.id != failed_run_id && run.scope_node_ids == vec![ordinary.clone()])
                .expect("actual ordinary child result");
            assert_eq!(completed.summary.passed, 1);
            assert_eq!(completed.summary.failed, 0);
            assert_eq!(completed.summary.cancelled, 0);
            assert!(completed.events.last().is_some_and(|event| {
                event.node_id == ordinary && event.state == StructuredNodeState::Passed
            }));
            assert!(
                !provider
                    .completed_runs
                    .iter()
                    .chain(provider.last_complete_run.iter())
                    .any(|run| run.id != failed_run_id && run.scope_node_ids.contains(&exact))
            );
        });
        let fresh = TestExplorerSelection {
            provider_id: provider_id.clone(),
            discovery_generation: snapshot.discovery_generation,
            node_id: exact.clone(),
            run_id: None,
        };
        cx.update(|window, cx| {
            RustTestExplorerDelegate
                .run(&fresh, &workspace.downgrade(), window, cx)
                .expect("fresh current Exact action")
        });
        let mut fresh_completed = false;
        for _ in 0..100 {
            cx.run_until_parked();
            fresh_completed = exact_finished.exists()
                && structured.read_with(cx, |store, _| {
                    let provider = store.state().provider(&provider_id).expect("provider");
                    provider
                        .completed_runs
                        .iter()
                        .chain(provider.last_complete_run.iter())
                        .any(|run| {
                            run.id != failed_run_id
                                && run.scope_node_ids == vec![exact.clone()]
                                && run.phase == StructuredRunPhase::Completed
                        })
                });
            if fresh_completed {
                break;
            }
            cx.background_executor
                .timer(Duration::from_millis(100))
                .await;
        }
        assert!(
            fresh_completed,
            "fresh opt-in action executes the actual rustdoc workload"
        );
        structured.read_with(cx, |store, _| {
            let provider = store.state().provider(&provider_id).expect("provider");
            let completed = provider
                .completed_runs
                .iter()
                .chain(provider.last_complete_run.iter())
                .find(|run| run.id != failed_run_id && run.scope_node_ids == vec![exact.clone()])
                .expect("actual fresh rustdoc child result");
            assert_eq!(completed.summary.passed, 1);
            assert_eq!(completed.summary.failed, 0);
            assert_eq!(completed.summary.cancelled, 0);
            assert!(completed.events.last().is_some_and(|event| {
                event.node_id == exact && event.state == StructuredNodeState::Passed
            }));
        });
        container.update_in(cx, |_, window, _| window.remove_window());
    }

    #[gpui::test]
    #[cfg(feature = "rust-test-actions")]
    async fn test_explorer_refresh_starts_cold_rust_discovery_without_passive_loops(
        cx: &mut TestAppContext,
    ) {
        crate::tests::init_test(cx);
        let fs = fs::FakeFs::new(cx.executor());
        let project = Project::test(fs, [], cx).await;
        let store = project.read_with(cx, |project, _| {
            project.structured_execution_store().clone()
        });
        let (multi_workspace, cx) = cx.add_window_view(|window, cx| {
            workspace::MultiWorkspace::test_new(project.clone(), window, cx)
        });
        let workspace = multi_workspace.read_with(cx, |workspace, _| workspace.workspace().clone());
        let async_cx = cx.update(|window, cx| window.to_async(cx));
        let panel = TestsPanel::load(workspace.downgrade(), async_cx)
            .await
            .expect("load Tests panel");
        cx.run_until_parked();
        let provider_id = StructuredProviderId(RUST_TEST_PROVIDER_ID.to_owned());
        store.update(cx, |store, _| {
            assert!(store.state().provider(&provider_id).is_none());
        });
        panel.update(cx, |panel, cx| {
            panel.active = true;
            panel.discover(cx);
        });
        cx.run_until_parked();
        let first = panel.read_with(cx, |panel, _| {
            let snapshot = panel
                .snapshots
                .iter()
                .find(|snapshot| snapshot.provider_id == provider_id)
                .expect("Rust snapshot");
            snapshot.discovery_generation
        });
        store.update(cx, |_, cx| {
            cx.emit(StructuredExecutionStoreEvent::Changed(provider_id.clone()));
        });
        cx.run_until_parked();
        panel.read_with(cx, |panel, _| {
            assert_eq!(
                panel
                    .snapshots
                    .iter()
                    .find(|snapshot| snapshot.provider_id == provider_id)
                    .expect("cached snapshot")
                    .discovery_generation,
                first
            );
        });
        panel.update(cx, |panel, cx| panel.discover(cx));
        cx.run_until_parked();
        panel.read_with(cx, |panel, _| {
            assert!(
                panel
                    .snapshots
                    .iter()
                    .find(|snapshot| snapshot.provider_id == provider_id)
                    .expect("refreshed snapshot")
                    .discovery_generation
                    > first
            );
        });
        let current_snapshots = panel.read_with(cx, |panel, _| panel.snapshots.clone());
        let current_roots = panel.read_with(cx, |panel, _| {
            panel
                .projection
                .snapshot
                .roots
                .iter()
                .map(|node| node.id.clone())
                .collect::<Vec<_>>()
        });
        panel.update(cx, |panel, cx| {
            let stale_generation = panel.host.start_refresh();
            let current_generation = panel.host.start_refresh();
            panel.apply_provider_snapshots(current_generation, current_snapshots.clone(), None, cx);
            panel.apply_provider_snapshots(
                stale_generation,
                vec![fake_provider(StructuredProviderStatus::Current)].into(),
                None,
                cx,
            );
            assert_eq!(panel.snapshots, current_snapshots);
        });
        cx.run_until_parked();
        panel.update(cx, |panel, cx| {
            assert_eq!(
                panel
                    .projection
                    .snapshot
                    .roots
                    .iter()
                    .map(|node| node.id.clone())
                    .collect::<Vec<_>>(),
                current_roots
            );
            let partial_generation = panel.host.start_refresh();
            panel.apply_provider_snapshots(
                partial_generation,
                vec![fake_provider(StructuredProviderStatus::Current)].into(),
                Some(anyhow::anyhow!("Rust discovery failed")),
                cx,
            );
        });
        cx.run_until_parked();
        panel.update(cx, |panel, cx| {
            assert!(matches!(panel.host.status(), LanguageToolTreeStatus::Partial(message) if message.contains("Rust discovery failed")));
            assert_eq!(panel.snapshots.first().expect("successful provider").provider_id.0, "web-tests");
            panel.filter.query = "renders".into();
            panel.rebuild_projection(cx);
        });
        cx.run_until_parked();
        panel.update(cx, |panel, cx| {
            assert!(matches!(panel.host.status(), LanguageToolTreeStatus::Partial(message) if message.contains("Rust discovery failed")));
            let recovered_generation = panel.host.start_refresh();
            panel.apply_provider_snapshots(recovered_generation, current_snapshots, None, cx);
        });
        cx.run_until_parked();
        panel.read_with(cx, |panel, _| {
            assert!(!matches!(
                panel.host.status(),
                LanguageToolTreeStatus::Partial(_)
            ))
        });
    }

    #[gpui::test]
    #[cfg(feature = "jvm-test-actions")]
    async fn test_explorer_java_readiness_preserves_unprojected_cached_cases(
        cx: &mut TestAppContext,
    ) {
        crate::tests::init_test(cx);
        let project = Project::test(fs::FakeFs::new(cx.executor()), [], cx).await;
        let (multi_workspace, cx) = cx.add_window_view(|window, cx| {
            workspace::MultiWorkspace::test_new(project.clone(), window, cx)
        });
        let workspace = multi_workspace.read_with(cx, |workspace, _| workspace.workspace().clone());
        let async_cx = cx.update(|window, cx| window.to_async(cx));
        let panel = TestsPanel::load(workspace.downgrade(), async_cx)
            .await
            .expect("load Tests panel");
        cx.run_until_parked();
        let store = project.read_with(cx, |project, _| {
            project.structured_execution_store().clone()
        });
        let mut cached = fake_provider(StructuredProviderStatus::Current);
        cached.provider_id = StructuredProviderId(JVM_JUNIT_PROVIDER_ID.to_owned());
        cached.discovery_generation = DiscoveryGeneration(7);
        store.update(cx, |store, _| {
            let project_generation = store.state().project_generation();
            store
                .state_mut()
                .apply_discovery(project_generation, cached.clone(), None)
                .expect("project-owned Java cases");
        });
        let (sender, receiver) = futures::channel::oneshot::channel();
        let delegate = Arc::new(QueuedDiscoveryDelegate(std::sync::Mutex::new(
            VecDeque::from([receiver]),
        )));
        panel.update(cx, |panel, cx| {
            panel.set_delegate(delegate.clone());
            panel.active = true;
            assert!(
                panel.snapshots.is_empty(),
                "projection has not caught up yet"
            );
            panel.request_java_discovery(false, cx);
        });
        assert_eq!(
            delegate.0.lock().expect("discovery queue lock").len(),
            1,
            "Java readiness must preserve existing project cases before panel projection"
        );
        panel.update(cx, |panel, cx| panel.refresh(cx));
        cx.run_until_parked();
        panel.read_with(cx, |panel, _| {
            let snapshot = panel.snapshots.first().expect("projected Java cases");
            assert_eq!(snapshot.provider_id, cached.provider_id);
            assert_eq!(snapshot.discovery_generation, cached.discovery_generation);
            assert_eq!(snapshot.nodes, cached.nodes);
        });
        panel.update(cx, |panel, cx| panel.request_java_discovery(true, cx));
        assert!(
            delegate.0.lock().expect("discovery queue lock").is_empty(),
            "forced discovery must still reach the provider"
        );
        sender
            .send(fake_provider(StructuredProviderStatus::Current))
            .expect("forced discovery remains live");
        cx.run_until_parked();
    }

    #[gpui::test]
    #[cfg(feature = "jvm-test-actions")]
    async fn test_explorer_discovers_again_when_java_and_jdtls_finish_loading(
        cx: &mut TestAppContext,
    ) {
        crate::tests::init_test(cx);
        let project = Project::test(fs::FakeFs::new(cx.executor()), [], cx).await;
        let languages = project.read_with(cx, |project, _| project.languages().clone());
        assert!(languages.language_for_name("Java").await.is_err());
        let (multi_workspace, cx) = cx.add_window_view(|window, cx| {
            workspace::MultiWorkspace::test_new(project.clone(), window, cx)
        });
        let workspace = multi_workspace.read_with(cx, |workspace, _| workspace.workspace().clone());
        let async_cx = cx.update(|window, cx| window.to_async(cx));
        let panel = TestsPanel::load(workspace.downgrade(), async_cx)
            .await
            .expect("load Tests panel");
        cx.run_until_parked();

        let (first_sender, first_receiver) = futures::channel::oneshot::channel();
        let (second_sender, second_receiver) = futures::channel::oneshot::channel();
        let delegate = Arc::new(QueuedDiscoveryDelegate(std::sync::Mutex::new(
            VecDeque::from([first_receiver, second_receiver]),
        )));
        panel.update(cx, |panel, cx| {
            panel.set_delegate(delegate.clone());
            panel.active = true;
            panel.discover(cx);
        });
        let mut first = fake_provider(StructuredProviderStatus::Current);
        first.discovery_generation = DiscoveryGeneration(1);
        first_sender
            .send(first)
            .expect("first discovery stays live");
        cx.run_until_parked();

        languages.register_test_language(language::LanguageConfig {
            name: "Java".into(),
            ..Default::default()
        });
        cx.run_until_parked();
        assert!(
            delegate.0.lock().expect("discovery queue lock").is_empty(),
            "Java language readiness should request one more discovery"
        );
        let mut second = fake_provider(StructuredProviderStatus::Current);
        second.discovery_generation = DiscoveryGeneration(2);
        second_sender
            .send(second)
            .expect("second discovery stays live");
        cx.run_until_parked();
        panel.read_with(cx, |panel, _| {
            assert_eq!(
                panel
                    .snapshots
                    .first()
                    .expect("rediscovered snapshot")
                    .discovery_generation,
                DiscoveryGeneration(2)
            );
        });

        let (third_sender, third_receiver) = futures::channel::oneshot::channel();
        delegate
            .0
            .lock()
            .expect("discovery queue lock")
            .push_back(third_receiver);
        panel.update(cx, |panel, _| {
            let mut cached = fake_provider(StructuredProviderStatus::Stale);
            cached.provider_id = StructuredProviderId(JVM_JUNIT_PROVIDER_ID.to_string());
            panel.snapshots = vec![cached].into();
        });
        let lsp_store = project.read_with(cx, |project, _| project.lsp_store());
        lsp_store.update(cx, |_, cx| {
            cx.emit(LspStoreEvent::LanguageServerAdded(
                LanguageServerId(1),
                LanguageServerName("kotlin-lsp".into()),
                None,
            ));
        });
        cx.run_until_parked();
        assert_eq!(
            delegate.0.lock().expect("discovery queue lock").len(),
            1,
            "an unrelated server must not restart Java discovery"
        );
        lsp_store.update(cx, |_, cx| {
            cx.emit(LspStoreEvent::LanguageServerAdded(
                LanguageServerId(2),
                LanguageServerName("jdtls".into()),
                None,
            ));
        });
        cx.run_until_parked();
        assert!(
            delegate.0.lock().expect("discovery queue lock").is_empty(),
            "JDTLS readiness must rediscover even when a cached report case is visible"
        );
        let mut third = fake_provider(StructuredProviderStatus::Current);
        third.discovery_generation = DiscoveryGeneration(3);
        third_sender
            .send(third)
            .expect("third discovery stays live");
        cx.run_until_parked();
        panel.read_with(cx, |panel, _| {
            assert_eq!(
                panel
                    .snapshots
                    .first()
                    .expect("JDTLS-ready snapshot")
                    .discovery_generation,
                DiscoveryGeneration(3)
            );
        });
    }

    #[gpui::test]
    #[cfg(feature = "jvm-test-actions")]
    async fn test_explorer_defers_java_discovery_until_active_run_finishes(
        cx: &mut TestAppContext,
    ) {
        crate::tests::init_test(cx);
        let project = Project::test(fs::FakeFs::new(cx.executor()), [], cx).await;
        let languages = project.read_with(cx, |project, _| project.languages().clone());
        let (multi_workspace, cx) = cx.add_window_view(|window, cx| {
            workspace::MultiWorkspace::test_new(project.clone(), window, cx)
        });
        let workspace = multi_workspace.read_with(cx, |workspace, _| workspace.workspace().clone());
        let async_cx = cx.update(|window, cx| window.to_async(cx));
        let panel = TestsPanel::load(workspace.downgrade(), async_cx)
            .await
            .expect("load Tests panel");
        cx.run_until_parked();

        let (first_sender, first_receiver) = futures::channel::oneshot::channel();
        let (second_sender, second_receiver) = futures::channel::oneshot::channel();
        let delegate = Arc::new(QueuedDiscoveryDelegate(std::sync::Mutex::new(
            VecDeque::from([first_receiver, second_receiver]),
        )));
        panel.update(cx, |panel, cx| {
            panel.set_delegate(delegate.clone());
            panel.active = true;
            panel.discover(cx);
        });
        let mut first = fake_provider(StructuredProviderStatus::Current);
        first.discovery_generation = DiscoveryGeneration(1);
        first.current_run = Some(StructuredRun::new(
            StructuredRunId("active-rust-run".into()),
            DiscoveryGeneration(1),
            vec![StructuredNodeId("browser-suite".into())],
        ));
        first_sender
            .send(first)
            .expect("first discovery stays live");
        cx.run_until_parked();

        let store = project.read_with(cx, |project, _| {
            project.structured_execution_store().clone()
        });
        store.update(cx, |store, _| {
            let project_generation = store.state().project_generation();
            let mut snapshot = fake_provider(StructuredProviderStatus::Current);
            snapshot.discovery_generation = DiscoveryGeneration(1);
            store
                .state_mut()
                .apply_discovery(project_generation, snapshot, None)
                .expect("register running provider");
            store
                .state_mut()
                .begin_run(
                    project_generation,
                    &StructuredProviderId("web-tests".into()),
                    StructuredRun::new(
                        StructuredRunId("active-rust-run".into()),
                        DiscoveryGeneration(1),
                        vec![StructuredNodeId("browser-suite".into())],
                    ),
                )
                .expect("begin running test");
        });

        languages.register_test_language(language::LanguageConfig {
            name: "Java".into(),
            ..Default::default()
        });
        cx.run_until_parked();
        panel.read_with(cx, |panel, _| assert!(panel.java_discovery_pending));
        assert_eq!(
            delegate.0.lock().expect("discovery queue lock").len(),
            1,
            "Java readiness must not displace a running test"
        );
        panel.update_in(cx, |panel, window, cx| {
            panel.set_active(false, window, cx);
            panel.host.mark_dirty();
            panel.set_active(true, window, cx);
        });
        assert_eq!(
            delegate.0.lock().expect("discovery queue lock").len(),
            1,
            "reopening Tests must not displace a running test"
        );

        store.update(cx, |store, cx| {
            let project_generation = store.state().project_generation();
            store
                .state_mut()
                .set_run_phase(
                    project_generation,
                    &StructuredProviderId("web-tests".into()),
                    &StructuredRunId("active-rust-run".into()),
                    project::structured_execution::StructuredRunPhase::Completed,
                    None,
                )
                .expect("finish running test");
            cx.emit(StructuredExecutionStoreEvent::Changed(
                StructuredProviderId("web-tests".into()),
            ));
        });
        cx.run_until_parked();
        assert!(delegate.0.lock().expect("discovery queue lock").is_empty());
        let mut second = fake_provider(StructuredProviderStatus::Current);
        second.discovery_generation = DiscoveryGeneration(2);
        second_sender
            .send(second)
            .expect("deferred discovery stays live");
        cx.run_until_parked();
        panel.read_with(cx, |panel, _| {
            assert!(!panel.java_discovery_pending);
            assert_eq!(
                panel
                    .snapshots
                    .first()
                    .expect("rediscovered snapshot")
                    .discovery_generation,
                DiscoveryGeneration(2)
            );
        });
    }

    #[gpui::test(iterations = 10)]
    async fn test_explorer_supersedes_inflight_discovery_before_projection(
        cx: &mut TestAppContext,
    ) {
        crate::tests::init_test(cx);
        let project = Project::test(fs::FakeFs::new(cx.executor()), [], cx).await;
        let (multi_workspace, cx) = cx.add_window_view(|window, cx| {
            workspace::MultiWorkspace::test_new(project.clone(), window, cx)
        });
        let workspace = multi_workspace.read_with(cx, |workspace, _| workspace.workspace().clone());
        let async_cx = cx.update(|window, cx| window.to_async(cx));
        let panel = TestsPanel::load(workspace.downgrade(), async_cx)
            .await
            .expect("load Tests panel");
        let (first_sender, first_receiver) = futures::channel::oneshot::channel();
        let (second_sender, second_receiver) = futures::channel::oneshot::channel();
        panel.update(cx, |panel, cx| {
            panel.set_delegate(Arc::new(QueuedDiscoveryDelegate(std::sync::Mutex::new(
                VecDeque::from([first_receiver, second_receiver]),
            ))));
            panel.discover(cx);
        });
        cx.run_until_parked();
        panel.read_with(cx, |panel, _| {
            assert!(panel.refresh_pending);
            assert!(panel.snapshots.is_empty());
        });

        panel.update(cx, |panel, cx| panel.discover(cx));
        cx.run_until_parked();
        let mut current = fake_provider(StructuredProviderStatus::Current);
        current.discovery_generation = DiscoveryGeneration(4);
        second_sender
            .send(current.clone())
            .expect("new discovery receiver stays active");
        cx.run_until_parked();
        panel.read_with(cx, |panel, _| {
            assert!(!panel.refresh_pending);
            assert_eq!(panel.snapshots.first(), Some(&current));
            assert!(!panel.projection.snapshot.roots.is_empty());
        });

        let stale = fake_provider(StructuredProviderStatus::Current);
        match first_sender.send(stale) {
            Ok(()) => cx.run_until_parked(),
            Err(_) => {}
        }
        panel.read_with(cx, |panel, _| {
            assert_eq!(panel.snapshots.first(), Some(&current));
            assert!(!panel.projection.snapshot.roots.is_empty());
        });
    }

    #[gpui::test]
    async fn structured_execution_production_projection_filters_and_rejects_superseded_inputs(
        cx: &mut TestAppContext,
    ) {
        crate::tests::init_test(cx);
        let project = Project::test(fs::FakeFs::new(cx.executor()), [], cx).await;
        let (container, cx) = cx.add_window_view(|window, cx| {
            workspace::MultiWorkspace::test_new(project.clone(), window, cx)
        });
        let workspace = container.read_with(cx, |container, _| container.workspace().clone());
        let async_cx = cx.update(|window, cx| window.to_async(cx));
        let panel = TestsPanel::load(workspace.downgrade(), async_cx)
            .await
            .expect("Tests panel");
        cx.run_until_parked();
        panel.update(cx, |panel, cx| {
            let generation = panel.host.start_refresh();
            panel.apply_provider_snapshots(
                generation,
                vec![fake_provider(StructuredProviderStatus::Current)].into(),
                None,
                cx,
            );
        });
        let store = project.read_with(cx, |project, _| {
            project.structured_execution_store().clone()
        });
        store.update(cx, |_, cx| {
            cx.emit(StructuredExecutionStoreEvent::Changed(
                StructuredProviderId("web-tests".into()),
            ))
        });
        cx.run_until_parked();
        panel.read_with(cx, |panel, _| {
            assert!(
                panel.projection.snapshot.roots.is_empty(),
                "invalidated queued results must not become visible"
            )
        });
        let (old_projection_generation, old_refresh_generation) = panel.update(cx, |panel, cx| {
            let old_generation = panel.host.start_refresh();
            panel.apply_provider_snapshots(
                old_generation,
                vec![fake_provider(StructuredProviderStatus::Current)].into(),
                None,
                cx,
            );
            let old_projection_generation = panel.projection_generation;
            assert!(
                panel.projection.snapshot.roots.is_empty(),
                "projection cannot run inline on the foreground"
            );
            let current_generation = panel.host.start_refresh();
            let mut current = fake_provider(StructuredProviderStatus::Current);
            current.provider_id = StructuredProviderId("current-provider".into());
            panel.apply_provider_snapshots(current_generation, vec![current].into(), None, cx);
            panel.set_status_filter(TestStatusFilter::Failed, cx);
            panel.set_status_filter(TestStatusFilter::Passed, cx);
            panel.apply_provider_snapshots(
                old_generation,
                vec![fake_provider(StructuredProviderStatus::Current)].into(),
                None,
                cx,
            );
            assert!(panel.projection.snapshot.roots.is_empty());
            (old_projection_generation, old_generation)
        });
        cx.run_until_parked();
        panel.update(cx, |panel, cx| {
            assert_eq!(
                panel.snapshots.first().expect("latest input").provider_id.0,
                "current-provider"
            );
            assert!(find_node(&panel.projection.snapshot.roots, "renders home").is_some());
            assert!(find_node(&panel.projection.snapshot.roots, "renders login").is_none());
            for (generation, refresh_generation) in [
                (old_projection_generation, panel.host.refresh_generation()),
                (panel.projection_generation, old_refresh_generation),
            ] {
                let stale = project_test_explorer(
                    &[fake_provider(StructuredProviderStatus::Current)],
                    &TestExplorerFilter::default(),
                );
                let tree = stale.snapshot.clone();
                panel.apply_projection(generation, refresh_generation, stale, tree, cx);
                assert!(find_node(&panel.projection.snapshot.roots, "renders home").is_some());
                assert!(
                    panel
                        .projection
                        .selections
                        .values()
                        .all(|selection| selection.provider_id.0 == "current-provider")
                );
            }
            panel.host.expand_all();
            panel.host.select(tree_id(
                &StructuredProviderId("current-provider".into()),
                &StructuredNodeId("renders-home".into()),
            ));
            panel.filter.query = "unmatched query".into();
            panel.rebuild_projection(cx);
            panel.filter.query = "home".into();
            panel.rebuild_projection(cx);
        });
        cx.run_until_parked();
        panel.read_with(cx, |panel, _| {
            assert!(find_node(&panel.projection.snapshot.roots, "renders home").is_some());
            assert_eq!(
                panel
                    .selected()
                    .expect("selection survives current projection")
                    .node_id
                    .0,
                "renders-home"
            );
            assert_eq!(panel.filter.query, "home");
        });
        cx.deactivate_window();
        container.update_in(cx, |_, window, _| window.remove_window());
    }

    #[gpui::test]
    async fn test_explorer_tree_host_preserves_selection_and_keyboard_behavior(
        cx: &mut TestAppContext,
    ) {
        let projection = project_test_explorer(
            &[fake_provider(StructuredProviderStatus::Current)],
            &TestExplorerFilter::default(),
        );
        let mut host = LanguageToolTreeHost::default();
        host.replace_snapshot(projection.snapshot);
        host.expand_all();
        host.select_first();
        let first = host.selected().cloned();
        host.select_next();
        assert_ne!(host.selected(), first.as_ref());
        host.select_parent();
        host.toggle_selected();
        assert!(host.selected().is_some());
        cx.background_executor
            .timer(std::time::Duration::from_millis(1))
            .await;
    }

    #[gpui::test]
    async fn structured_execution_foreground_budget(cx: &mut TestAppContext) {
        use std::sync::atomic::Ordering;
        const NODE_LIMIT: usize = 10_000;
        const FOREGROUND_BUDGET: Duration = Duration::from_millis(250);
        crate::tests::init_test(cx);
        let project = Project::test(fs::FakeFs::new(cx.executor()), [], cx).await;
        let provider_id = StructuredProviderId("scale-provider".into());
        let root_id = StructuredNodeId("scale-root".into());
        let run_id = StructuredRunId("scale-run".into());
        let mut nodes = Vec::with_capacity(NODE_LIMIT);
        nodes.push(StructuredNode {
            experimental_feature: None,
            experimental_feature_enabled: None,
            debug_capability: None,
            id: root_id.clone(),
            parent_id: None,
            label: "Scale tests".into(),
            kind: StructuredNodeKind::Provider,
            path: None,
        });
        nodes.extend((1..NODE_LIMIT).map(|index| StructuredNode {
            experimental_feature: None,
            experimental_feature_enabled: None,
            debug_capability: None,
            id: StructuredNodeId(format!("scale-case-{index:05}")),
            parent_id: Some(root_id.clone()),
            label: format!("Scale case {index:05}"),
            kind: StructuredNodeKind::Case,
            path: None,
        }));
        let provider = StructuredProviderSnapshot::discovery(
            provider_id.clone(),
            DiscoveryGeneration(1),
            StructuredProviderStatus::Current,
            nodes,
        );
        let store = project.read_with(cx, |project, _| {
            project.structured_execution_store().clone()
        });
        store.update(cx, |store, _| {
            let state = store.state_mut();
            let generation = state.project_generation();
            state
                .apply_discovery(generation, provider, None)
                .expect("discovery");
            state
                .begin_run(
                    generation,
                    &provider_id,
                    StructuredRun::new(run_id.clone(), DiscoveryGeneration(1), vec![root_id]),
                )
                .expect("run");
            for index in 1..NODE_LIMIT {
                state
                    .apply_event(
                        generation,
                        &provider_id,
                        &run_id,
                        StructuredExecutionEvent {
                            sequence: (index - 1) as u64,
                            node_id: StructuredNodeId(format!("scale-case-{index:05}")),
                            state: StructuredNodeState::Passed,
                            duration_millis: Some(1),
                            message: None,
                            location: None,
                        },
                        None,
                    )
                    .expect("event");
            }
        });
        let (container, cx) = cx.add_window_view(|window, cx| {
            workspace::MultiWorkspace::test_new(project.clone(), window, cx)
        });
        let workspace = container.read_with(cx, |container, _| container.workspace().clone());
        let async_cx = cx.update(|window, cx| window.to_async(cx));
        let panel = TestsPanel::load(workspace.downgrade(), async_cx)
            .await
            .expect("Tests panel");
        cx.run_until_parked();
        panel.update(cx, |panel, _| panel.active = true);
        for phase in ["refresh", "query", "results", "status"] {
            panel.update(cx, |panel, _| {
                panel.projection_timings = TestsProjectionTimings::default()
            });
            let gate = if phase == "query" {
                let (sender, receiver) = futures::channel::oneshot::channel();
                panel.update(cx, |panel, _| {
                    panel.projection_timings.background_gate = Some(receiver)
                });
                Some(sender)
            } else {
                None
            };
            let started = Instant::now();
            let scheduling_started = Instant::now();
            match phase {
                "refresh" => panel.update(cx, |panel, cx| panel.refresh(cx)),
                "query" => {
                    let editor = panel
                        .read_with(cx, |panel, cx| panel.query_input.read(cx).editor().clone());
                    cx.update(|window, cx| editor.set_text("case", window, cx));
                }
                "results" => store.update(cx, |store, cx| {
                    let state = store.state_mut();
                    state
                        .apply_event(
                            state.project_generation(),
                            &provider_id,
                            &run_id,
                            StructuredExecutionEvent {
                                sequence: (NODE_LIMIT - 1) as u64,
                                node_id: StructuredNodeId(format!(
                                    "scale-case-{:05}",
                                    NODE_LIMIT - 1
                                )),
                                state: StructuredNodeState::Failed,
                                duration_millis: Some(42),
                                message: Some("changed result".into()),
                                location: None,
                            },
                            None,
                        )
                        .expect("changed result");
                    cx.emit(StructuredExecutionStoreEvent::Changed(provider_id.clone()));
                }),
                "status" => panel.update(cx, |panel, cx| {
                    panel.set_status_filter(TestStatusFilter::Passed, cx)
                }),
                _ => unreachable!(),
            }
            let scheduling = scheduling_started.elapsed();
            cx.background_executor.timer(Duration::from_millis(1)).await;
            cx.run_until_parked();
            let mut pending_paint = Duration::ZERO;
            if let Some(gate) = gate {
                panel.read_with(cx, |panel, _| {
                    assert_eq!(panel.projection_timings.application, Duration::ZERO);
                    assert_eq!(panel.host.visible_rows().len(), NODE_LIMIT);
                    assert_eq!(panel.filter.query, "case");
                });
                let pending_started = Instant::now();
                cx.draw(
                    gpui::point(px(0.0), px(0.0)),
                    gpui::size(px(360.0), px(720.0)),
                    |_, _| panel.clone().into_any_element(),
                );
                pending_paint = pending_started.elapsed();
                gate.send(()).expect("release background projection");
                cx.background_executor.timer(Duration::from_millis(1)).await;
                cx.run_until_parked();
            }
            let painting_started = Instant::now();
            let row_count = panel.update(cx, |panel, cx| {
                assert_eq!(
                    panel.snapshots.first().expect("provider").nodes.len(),
                    NODE_LIMIT
                );
                panel.host.expand_all();
                panel.host.select_last();
                panel.host.reveal_selection(ScrollStrategy::Bottom);
                cx.notify();
                panel.host.visible_rows().len()
            });
            cx.draw(
                gpui::point(px(0.0), px(0.0)),
                gpui::size(px(360.0), px(720.0)),
                |_, _| panel.clone().into_any_element(),
            );
            let painting = painting_started.elapsed();
            let total = started.elapsed();
            let (foreground, background) = panel.read_with(cx, |panel, _| {
                let timings = &panel.projection_timings;
                let scroll = panel.host.scroll_handle().0.borrow();
                let measured = scroll.last_item_size.expect("painted list");
                assert!(measured.item.height > px(0.0) && measured.item.height <= px(720.0));
                let bottom = measured.item.height - measured.contents.height;
                assert!((scroll.base_handle.offset().y - bottom).abs() <= px(1.0));
                (
                    scheduling
                        + timings.refresh
                        + timings.admission
                        + timings.application
                        + pending_paint
                        + painting,
                    Duration::from_nanos(timings.background_nanoseconds.load(Ordering::SeqCst)),
                )
            });
            assert_eq!(
                row_count,
                if phase == "status" {
                    NODE_LIMIT - 1
                } else {
                    NODE_LIMIT
                }
            );
            assert!(
                foreground <= FOREGROUND_BUDGET,
                "{phase}: production foreground work took {foreground:?}, exceeding {FOREGROUND_BUDGET:?}; background={background:?}, total={total:?}"
            );
            eprintln!(
                "structured-execution-production-budget phase={phase} input_nodes={NODE_LIMIT} rows={row_count} foreground_us={} background_us={} total_ms={}",
                foreground.as_micros(),
                background.as_micros(),
                total.as_millis()
            );
        }
        cx.deactivate_window();
        container.update_in(cx, |_, window, _| window.remove_window());
    }

    fn find_node<'a>(nodes: &'a [LanguageToolNode], label: &str) -> Option<&'a LanguageToolNode> {
        for node in nodes {
            if node.label == label {
                return Some(node);
            }
            if let Some(found) = find_node(&node.children, label) {
                return Some(found);
            }
        }
        None
    }

    fn collect_ids(nodes: &[LanguageToolNode]) -> Vec<LanguageToolNodeId> {
        let mut ids = Vec::new();
        for node in nodes {
            ids.push(node.id.clone());
            ids.extend(collect_ids(&node.children));
        }
        ids
    }
}
