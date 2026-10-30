use std::{
    collections::{BTreeMap, HashMap, HashSet},
    io::Read as _,
    path::{Path, PathBuf},
    time::Duration,
};

use anyhow::{Context as _, Result, bail, ensure};
use fs::{Fs, MTime};
use futures::{
    FutureExt as _, StreamExt as _,
    channel::{mpsc, oneshot},
    future::{AbortHandle, Abortable, Either, select},
};
use gpui::{
    AppContext as _, AsyncApp, Context, Entity, Subscription, Task, TaskExt as _, WeakEntity,
};
use language::{LanguageName, ParseStatus, RunnableRange};
use path::PathStyle;
use quick_xml::{
    Reader,
    events::{BytesStart, Event},
};
use rpc::{AnyProtoClient, TypedEnvelope, proto};
use sha2::{Digest as _, Sha256};
use task::{
    StructuredTaskHandle, StructuredTaskLifecycleEvent, StructuredTaskState, StructuredTerminalId,
    TaskId, TaskTemplate,
};
use util::rel_path::RelPath;

use crate::{
    ProjectPath, WorktreeId,
    buffer_store::BufferStore,
    structured_execution::{
        DiscoveryGeneration, StructuredExecutionEvent, StructuredExecutionStore,
        StructuredExecutionStoreEvent, StructuredNode, StructuredNodeId, StructuredNodeKind,
        StructuredNodeState, StructuredProviderId, StructuredProviderSnapshot,
        StructuredProviderStatus, StructuredRun, StructuredRunId, StructuredRunPhase,
        provider_status_to_proto,
    },
    task_inventory::jvm_build_entry_is_available,
    trusted_worktrees::{TrustedWorktrees, TrustedWorktreesEvent},
    worktree_store::{WorktreeStore, WorktreeStoreEvent},
};

pub const JVM_JUNIT_PROVIDER_ID: &str = "jvm-junit";
pub const MAX_JUNIT_REPORT_BYTES: usize = 4 * 1024 * 1024;
pub const MAX_JUNIT_CASES: usize = 3_000;
pub const MAX_JUNIT_REPORT_FILES: usize = 128;
pub const MAX_JUNIT_DISCOVERY_BYTES: usize = 16 * 1024 * 1024;
pub const MAX_JUNIT_MODULES: usize = 128;
pub const MAX_JUNIT_SOURCE_FILES: usize = 128;
pub const MAX_JUNIT_SOURCE_BYTES: usize = 1024 * 1024;
pub const MAX_JUNIT_SOURCE_TOTAL_BYTES: usize = 16 * 1024 * 1024;
pub const JVM_TEST_DISCOVERY_PROTOCOL_VERSION: u32 = 1;
pub const JVM_TEST_BOUNDED_ACTION_PROTOCOL_VERSION: u32 = 2;
const MAX_REMOTE_JVM_WORKTREES: usize = 128;
const MAX_REMOTE_JVM_RUNS: usize = 32;
const MAX_REMOTE_JVM_FIELD_BYTES: usize = 1024;
const MAX_REMOTE_JVM_PLAN_BYTES: usize = 8192;
const MAX_JUNIT_XML_DEPTH: usize = 32;
const MAX_JUNIT_FIELD_BYTES: usize = 1_024;
const JVM_TASK_SPAWN_FAILURE: &str =
    "The Java test Task failed to start; inspect the Task terminal and project-host logs";

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct JunitCase {
    pub class_name: String,
    pub display_name: String,
    pub state: StructuredNodeState,
    pub message: Option<String>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct JunitReport {
    pub scope_path: ProjectPath,
    pub module_path: String,
    pub cases: Vec<JunitCase>,
    pub version: Option<JunitReportVersion>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct JunitReportVersion {
    relative_path: PathBuf,
    inode: u64,
    mtime: MTime,
    size: u64,
    digest: [u8; 32],
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct JvmTestSelector {
    pub scope_path: ProjectPath,
    pub module_path: String,
    pub class_name: String,
    pub method_name: String,
}

#[derive(Clone, Debug, PartialEq)]
pub struct JvmSelectedTestPlan {
    pub worktree_id: WorktreeId,
    pub discovery_generation: DiscoveryGeneration,
    pub node_id: StructuredNodeId,
    pub selector: JvmTestSelector,
    pub build_tool: JvmTestBuildTool,
    pub template: TaskTemplate,
}

#[derive(Clone, Debug, PartialEq)]
pub struct JvmRemoteTestActionPlan {
    pub run_id: StructuredRunId,
    pub worktree_id: WorktreeId,
    pub discovery_generation: DiscoveryGeneration,
    pub node_id: StructuredNodeId,
    pub template: TaskTemplate,
    client_generation: u64,
    report_maximum_bytes: Option<usize>,
    cancellation: Option<std::sync::Arc<BoundedRemotePlanCancellation>>,
}

struct BoundedRemotePlanCancellation {
    request: proto::UpdateJvmTestRun,
    sender: mpsc::UnboundedSender<proto::UpdateJvmTestRun>,
    armed: std::sync::atomic::AtomicBool,
}

impl std::fmt::Debug for BoundedRemotePlanCancellation {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("BoundedRemotePlanCancellation")
            .field("run_id", &self.request.run_id)
            .finish_non_exhaustive()
    }
}

impl PartialEq for BoundedRemotePlanCancellation {
    fn eq(&self, other: &Self) -> bool {
        self.request == other.request
    }
}

impl Drop for BoundedRemotePlanCancellation {
    fn drop(&mut self) {
        if self.armed.load(std::sync::atomic::Ordering::Relaxed) {
            if let Err(error) = self.sender.unbounded_send(self.request.clone()) {
                log::debug!("Bounded remote Java plan cleanup unavailable: {error}");
            }
        }
    }
}

impl JvmRemoteTestActionPlan {
    pub fn report_maximum_bytes(&self) -> Option<usize> {
        self.report_maximum_bytes
    }
}

struct AuthorizedRemoteJvmRun {
    plan: JvmSelectedTestPlan,
    previous_report: Option<JunitReportVersion>,
    task_id: Option<TaskId>,
    pending_cancel_confirmation: bool,
    report_maximum_bytes: Option<usize>,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum JvmTestBuildTool {
    Maven,
    Gradle,
}

struct JvmModuleScope {
    root: std::sync::Arc<Path>,
    manifest: ProjectPath,
    module_path: String,
    build_tool: JvmTestBuildTool,
}

pub struct JvmTestProviderStore {
    remote: Option<(u64, AnyProtoClient)>,
    worktree_store: Entity<WorktreeStore>,
    buffer_store: WeakEntity<BufferStore>,
    structured_store: Entity<StructuredExecutionStore>,
    generation: u64,
    next_run_id: u64,
    closed: bool,
    disconnected: bool,
    selectors: HashMap<StructuredNodeId, JvmTestSelector>,
    run_handles: HashMap<StructuredRunId, StructuredTaskHandle>,
    current_remote_discovery: Option<DiscoveryGeneration>,
    active_remote_request_id: Option<u64>,
    active_remote_request_abort: Option<AbortHandle>,
    active_remote_requests: HashMap<(proto::PeerId, u64), AbortHandle>,
    cancelled_remote_requests: HashSet<(proto::PeerId, u64)>,
    authorized_remote_runs: HashMap<(proto::PeerId, StructuredRunId), AuthorizedRemoteJvmRun>,
    remote_lifecycle_tx: Option<mpsc::UnboundedSender<proto::UpdateJvmTestRun>>,
    _remote_lifecycle_task: Task<()>,
    #[cfg(any(test, feature = "test-support"))]
    discovery_reply_gate: Option<(
        futures::channel::oneshot::Sender<()>,
        futures::channel::oneshot::Receiver<()>,
        futures::channel::oneshot::Sender<()>,
    )>,
    #[cfg(any(test, feature = "test-support"))]
    action_reply_gate: Option<(
        futures::channel::oneshot::Sender<()>,
        futures::channel::oneshot::Receiver<()>,
    )>,
    _subscriptions: Vec<Subscription>,
}

impl JvmTestProviderStore {
    pub fn init(client: &AnyProtoClient) {
        client.add_entity_request_handler(Self::handle_get_discovery);
        client.add_entity_request_handler(Self::handle_cancel_discovery);
        client.add_entity_request_handler(Self::handle_resolve_action);
        client.add_entity_request_handler(Self::handle_update_run);
    }

    async fn handle_get_discovery(
        this: Entity<Self>,
        envelope: TypedEnvelope<proto::GetJvmTestDiscovery>,
        mut cx: AsyncApp,
    ) -> Result<proto::JvmTestDiscoveryResponse> {
        let request = envelope.payload;
        if request.protocol_version != JVM_TEST_DISCOVERY_PROTOCOL_VERSION {
            return Ok(proto::JvmTestDiscoveryResponse {
                protocol_version: JVM_TEST_DISCOVERY_PROTOCOL_VERSION,
                discovery_generation: 0,
                request_id: request.request_id,
                status: proto::StructuredProviderStatus::Mismatch as i32,
                diagnostic: Some("Java test discovery protocol mismatch".to_string()),
            });
        }
        ensure!(
            request.worktree_ids.len() <= MAX_REMOTE_JVM_WORKTREES,
            "Java test discovery worktree scope exceeds the supported limit"
        );
        let peer_id = envelope.original_sender_id.unwrap_or(envelope.sender_id);
        let key = (peer_id, request.request_id);
        let allowed = request
            .worktree_ids
            .into_iter()
            .map(WorktreeId::from_proto)
            .collect();
        let (abort_handle, abort_registration) = AbortHandle::new_pair();
        let proceed = this.update(&mut cx, |store, _| {
            ensure!(
                store.remote.is_none(),
                "Java test discovery reached a client proxy"
            );
            if store.cancelled_remote_requests.remove(&key) {
                return Ok(false);
            }
            store.active_remote_requests.insert(key, abort_handle);
            Ok(true)
        })?;
        ensure!(proceed, "Java test discovery was cancelled");
        let refresh = this.update(&mut cx, |store, cx| store.refresh_scoped(Some(allowed), cx));
        let result = Abortable::new(refresh, abort_registration).await;
        this.update(&mut cx, |store, _| {
            store.active_remote_requests.remove(&key);
            store.cancelled_remote_requests.remove(&key);
        });
        let snapshot =
            result.map_err(|_| anyhow::anyhow!("Java test discovery was cancelled"))??;
        #[cfg(any(test, feature = "test-support"))]
        if let Some((started, release, completed)) =
            this.update(&mut cx, |store, _| store.discovery_reply_gate.take())
        {
            started
                .send(())
                .map_err(|_| anyhow::anyhow!("Java discovery reply observer was dropped"))?;
            release
                .await
                .map_err(|_| anyhow::anyhow!("Java discovery reply release was dropped"))?;
            if completed.send(()).is_err() {
                log::debug!("Java discovery reply completion observer was dropped");
            }
        }
        Ok(proto::JvmTestDiscoveryResponse {
            protocol_version: JVM_TEST_DISCOVERY_PROTOCOL_VERSION,
            discovery_generation: snapshot.discovery_generation.0,
            request_id: request.request_id,
            status: provider_status_to_proto(snapshot.status),
            diagnostic: snapshot.diagnostic,
        })
    }

    async fn handle_cancel_discovery(
        this: Entity<Self>,
        envelope: TypedEnvelope<proto::CancelJvmTestDiscovery>,
        mut cx: AsyncApp,
    ) -> Result<proto::Ack> {
        let peer_id = envelope.original_sender_id.unwrap_or(envelope.sender_id);
        let key = (peer_id, envelope.payload.request_id);
        this.update(&mut cx, |store, _| {
            if let Some(abort) = store.active_remote_requests.remove(&key) {
                abort.abort();
            } else if store.cancelled_remote_requests.len() < MAX_REMOTE_JVM_WORKTREES {
                store.cancelled_remote_requests.insert(key);
            } else {
                log::warn!("Ignored excess pending Java test discovery cancellations");
            }
        });
        Ok(proto::Ack {})
    }

    async fn handle_resolve_action(
        this: Entity<Self>,
        envelope: TypedEnvelope<proto::ResolveJvmTestAction>,
        mut cx: AsyncApp,
    ) -> Result<proto::JvmTestActionPlanResponse> {
        let request = envelope.payload;
        validate_jvm_action_protocol(
            request.protocol_version,
            request.report_maximum_bytes.is_some(),
        )?;
        let report_maximum_bytes = request
            .report_maximum_bytes
            .map(|maximum| {
                let maximum =
                    usize::try_from(maximum).context("Java report byte limit is unsupported")?;
                ensure!(
                    (1..=MAX_JUNIT_DISCOVERY_BYTES).contains(&maximum),
                    "Java report byte limit is unsupported"
                );
                Ok::<_, anyhow::Error>(maximum)
            })
            .transpose()?;
        let requested_run_id = request.requested_run_id.map(StructuredRunId);
        ensure!(
            report_maximum_bytes.is_some() == requested_run_id.is_some(),
            "Bounded Java report requests must identify their private run reservation"
        );
        if let Some(run_id) = &requested_run_id {
            let suffix = run_id
                .0
                .strip_prefix("jvm-bounded-")
                .context("Bounded Java run identity is invalid")?;
            ensure!(
                suffix.len() == 32 && suffix.bytes().all(|byte| byte.is_ascii_hexdigit()),
                "Bounded Java run identity is invalid"
            );
        }
        let peer_id = envelope.original_sender_id.unwrap_or(envelope.sender_id);
        validate_remote_jvm_field("Java test node ID", &request.node_id)?;
        if let Some(failed_run_id) = &request.failed_run_id {
            validate_remote_jvm_field("Java test failed run ID", failed_run_id)?;
        }
        ensure!(
            request.worktree_ids.len() <= MAX_REMOTE_JVM_WORKTREES,
            "Java test action worktree scope exceeds the supported limit"
        );
        let allowed = request
            .worktree_ids
            .into_iter()
            .map(WorktreeId::from_proto)
            .collect::<HashSet<_>>();
        let node_id = StructuredNodeId(request.node_id);
        let generation = DiscoveryGeneration(request.discovery_generation);
        let failed_run_id = request.failed_run_id.map(StructuredRunId);
        let plan = this.update(&mut cx, |store, cx| {
            ensure!(
                store.remote.is_none(),
                "Java test action reached a client proxy"
            );
            if report_maximum_bytes.is_some() {
                store.ensure_bounded_remote_run_idle(cx)?;
            }
            let plan = match &failed_run_id {
                Some(run_id) => store.plan_failed_case(&node_id, generation, run_id, cx)?,
                None => store.plan_selected_case(&node_id, generation, cx)?,
            };
            ensure!(
                allowed.contains(&plan.worktree_id),
                "The selected Java test is outside the requested worktree scope"
            );
            store.ensure_host_action_trusted(plan.worktree_id, cx)?;
            if let Some(run_id) = &requested_run_id {
                ensure!(
                    store.authorized_remote_runs.len() < MAX_REMOTE_JVM_RUNS
                        && !store
                            .authorized_remote_runs
                            .contains_key(&(peer_id, run_id.clone())),
                    "Bounded Java plan is duplicate or reservation capacity is unavailable"
                );
                store.authorized_remote_runs.insert(
                    (peer_id, run_id.clone()),
                    AuthorizedRemoteJvmRun {
                        plan: plan.clone(),
                        previous_report: None,
                        task_id: None,
                        pending_cancel_confirmation: false,
                        report_maximum_bytes,
                    },
                );
            }
            Ok::<_, anyhow::Error>(plan)
        })?;
        let previous_report = this
            .update(&mut cx, |store, cx| {
                store.capture_report_version_with_limit(
                    &plan,
                    report_maximum_bytes.unwrap_or(MAX_JUNIT_DISCOVERY_BYTES),
                    cx,
                )
            })
            .await;
        let previous_report = match previous_report {
            Ok(version) => version,
            Err(error) => {
                if let Some(run_id) = &requested_run_id {
                    this.update(&mut cx, |store, _| {
                        store
                            .authorized_remote_runs
                            .remove(&(peer_id, run_id.clone()));
                    });
                }
                return Err(error);
            }
        };
        #[cfg(any(test, feature = "test-support"))]
        if let Some((started, release)) =
            this.update(&mut cx, |store, _| store.action_reply_gate.take())
        {
            started
                .send(())
                .map_err(|_| anyhow::anyhow!("Java action reply observer was dropped"))?;
            release
                .await
                .map_err(|_| anyhow::anyhow!("Java action reply release was dropped"))?;
        }
        let response = this.update(&mut cx, |store, cx| {
            if let Some(run_id) = &requested_run_id {
                ensure!(
                    store
                        .authorized_remote_runs
                        .get(&(peer_id, run_id.clone()))
                        .is_some_and(|run| run.previous_report.is_none() && run.task_id.is_none()),
                    "Bounded Java inspection cancelled before its plan was granted"
                );
            }
            if report_maximum_bytes.is_some() {
                store.ensure_bounded_remote_run_idle(cx)?;
            }
            let current = match &failed_run_id {
                Some(run_id) => store.plan_failed_case(&node_id, generation, run_id, cx)?,
                None => store.plan_selected_case(&node_id, generation, cx)?,
            };
            ensure!(
                current == plan,
                "The selected Java test changed before authorization"
            );
            store.ensure_host_action_trusted(current.worktree_id, cx)?;
            if store.authorized_remote_runs.len() >= MAX_REMOTE_JVM_RUNS {
                store
                    .authorized_remote_runs
                    .retain(|_, run| !run.pending_cancel_confirmation);
            }
            ensure!(
                requested_run_id.is_some()
                    || store.authorized_remote_runs.len() < MAX_REMOTE_JVM_RUNS,
                "Too many pending remote Java test runs"
            );
            let template_json = serde_json::to_string(&plan.template)?;
            ensure!(
                template_json.len() <= MAX_REMOTE_JVM_PLAN_BYTES,
                "Remote Java test action plan exceeds the supported limit"
            );
            store.next_run_id = store.next_run_id.wrapping_add(1);
            let run_id = requested_run_id.clone().unwrap_or_else(|| {
                StructuredRunId(format!("jvm-junit-remote-run-{}", store.next_run_id))
            });
            store.authorized_remote_runs.insert(
                (peer_id, run_id.clone()),
                AuthorizedRemoteJvmRun {
                    plan: plan.clone(),
                    previous_report: Some(previous_report),
                    task_id: None,
                    pending_cancel_confirmation: false,
                    report_maximum_bytes,
                },
            );
            Ok(proto::JvmTestActionPlanResponse {
                protocol_version: request.protocol_version,
                discovery_generation: generation.0,
                node_id: node_id.0,
                run_id: run_id.0,
                worktree_id: plan.worktree_id.to_proto(),
                template_json,
                report_maximum_bytes: report_maximum_bytes.map(|maximum| maximum as u64),
            })
        });
        if response.is_err() {
            if let Some(run_id) = &requested_run_id {
                this.update(&mut cx, |store, _| {
                    store
                        .authorized_remote_runs
                        .remove(&(peer_id, run_id.clone()));
                });
            }
        }
        response
    }

    fn ensure_bounded_remote_run_idle(&self, cx: &gpui::App) -> Result<()> {
        ensure!(
            !self.has_active_runs()
                && !self
                    .structured_store
                    .read(cx)
                    .state()
                    .provider(&StructuredProviderId(JVM_JUNIT_PROVIDER_ID.into()))
                    .and_then(|provider| provider.current_run.as_ref())
                    .is_some_and(|run| !run.phase.is_terminal()),
            "Another Java test run is active; bounded diagnosis cannot supersede it"
        );
        Ok(())
    }

    fn ensure_host_action_trusted(
        &self,
        worktree_id: WorktreeId,
        cx: &mut Context<Self>,
    ) -> Result<()> {
        ensure!(
            self.worktree_store
                .read(cx)
                .visible_worktrees(cx)
                .any(|worktree| worktree.read(cx).id() == worktree_id),
            "The selected Java test worktree is no longer visible; refresh Tests"
        );
        let trusted_worktrees = TrustedWorktrees::try_get_global(cx)
            .context("Project host trust is unavailable; reconnect before running Java tests")?;
        let trusted = trusted_worktrees.update(cx, |trusted, cx| {
            trusted.can_trust(&self.worktree_store, worktree_id, cx)
        });
        ensure!(
            trusted,
            "Trust the project host worktree before running a Java test; then refresh Tests"
        );
        Ok(())
    }

    async fn handle_update_run(
        this: Entity<Self>,
        envelope: TypedEnvelope<proto::UpdateJvmTestRun>,
        mut cx: AsyncApp,
    ) -> Result<proto::Ack> {
        let request = envelope.payload;
        ensure_jvm_test_protocol(request.protocol_version)?;
        ensure!(
            request.worktree_ids.len() <= MAX_REMOTE_JVM_WORKTREES,
            "Java test lifecycle worktree scope exceeds the supported limit"
        );
        let allowed = request
            .worktree_ids
            .iter()
            .copied()
            .map(WorktreeId::from_proto)
            .collect::<HashSet<_>>();
        let event = remote_jvm_lifecycle_from_proto(&request)?;
        let run_id = StructuredRunId(request.run_id);
        let node_id = StructuredNodeId(request.node_id);
        let peer_id = envelope.original_sender_id.unwrap_or(envelope.sender_id);
        let key = (peer_id, run_id.clone());
        this.update(&mut cx, |store, cx| {
            let worktree_id = store
                .authorized_remote_runs
                .get(&key)
                .context("Unknown or expired remote Java test run")?
                .plan
                .worktree_id;
            if let Err(error) = store.ensure_host_action_trusted(worktree_id, cx) {
                store.authorized_remote_runs.remove(&key);
                return Err(error);
            }
            let authorization = store
                .authorized_remote_runs
                .get(&key)
                .context("Unknown or expired remote Java test run")?;
            let bounded = authorization.report_maximum_bytes.is_some();
            if request.abandon_unlaunched
                && bounded
                && authorization.task_id.is_none()
                && matches!(
                    event.state,
                    StructuredTaskState::Cancelled {
                        termination_confirmed: true,
                        ..
                    }
                )
            {
                ensure!(
                    authorization.plan.discovery_generation.0 == request.discovery_generation
                        && authorization.plan.node_id == node_id
                        && allowed.contains(&authorization.plan.worktree_id),
                    "Remote Java plan cancellation scope changed"
                );
                store.authorized_remote_runs.remove(&key);
                return Ok(proto::Ack {});
            }
            ensure!(
                !request.abandon_unlaunched,
                "Only an unlaunched bounded Java reservation can be abandoned"
            );
            if bounded && authorization.task_id.is_none() {
                store.ensure_bounded_remote_run_idle(cx)?;
            }
            let authorization = store
                .authorized_remote_runs
                .get_mut(&key)
                .context("Unknown or expired remote Java test run")?;
            match &authorization.task_id {
                Some(task_id) => ensure!(
                    task_id == &event.task_id,
                    "Remote Java test lifecycle Task identity mismatch"
                ),
                None => {
                    ensure!(
                        matches!(&event.state, StructuredTaskState::Queued),
                        "Remote Java test lifecycle did not begin with a queued state"
                    );
                    authorization.task_id = Some(event.task_id.clone());
                }
            }
            ensure!(
                authorization.plan.discovery_generation.0 == request.discovery_generation
                    && authorization.plan.node_id == node_id
                    && allowed.contains(&authorization.plan.worktree_id)
                    && store.generation == request.discovery_generation,
                "Remote Java test lifecycle identity or worktree scope mismatch"
            );
            let plan = authorization.plan.clone();
            let previous_report = authorization
                .previous_report
                .clone()
                .context("Bounded Java baseline inspection is not complete")?;
            let maximum_bytes = authorization
                .report_maximum_bytes
                .unwrap_or(MAX_JUNIT_DISCOVERY_BYTES);
            let provider_id = StructuredProviderId(JVM_JUNIT_PROVIDER_ID.to_string());
            let current_run = store
                .structured_store
                .read(cx)
                .state()
                .provider(&provider_id)
                .and_then(|provider| provider.current_run.as_ref())
                .map(|run| run.id.clone());
            if current_run.as_ref() != Some(&run_id) {
                ensure!(
                    matches!(&event.state, StructuredTaskState::Queued),
                    "Remote Java test lifecycle did not begin with a queued state"
                );
                store.structured_store.update(cx, |structured, cx| {
                    let project_generation = structured.state().project_generation();
                    structured.state_mut().begin_run(
                        project_generation,
                        &provider_id,
                        StructuredRun::new(
                            run_id.clone(),
                            plan.discovery_generation,
                            vec![plan.node_id.clone()],
                        ),
                    )?;
                    cx.emit(StructuredExecutionStoreEvent::Changed(provider_id));
                    anyhow::Ok(())
                })?;
            }
            store.process_run_event(&run_id, &plan, &previous_report, &event, maximum_bytes, cx)?;
            if matches!(
                &event.state,
                StructuredTaskState::Cancelled {
                    termination_confirmed: false,
                    ..
                }
            ) {
                if let Some(authorization) = store.authorized_remote_runs.get_mut(&key) {
                    authorization.pending_cancel_confirmation = true;
                }
            } else if event.state.is_terminal() {
                store.authorized_remote_runs.remove(&key);
            }
            Ok(proto::Ack {})
        })
    }

    pub fn selector(&self, node_id: &StructuredNodeId) -> Option<&JvmTestSelector> {
        self.selectors.get(node_id)
    }

    pub fn is_remote(&self) -> bool {
        self.remote.is_some()
    }

    pub fn plan_remote_action(
        &self,
        node_id: &StructuredNodeId,
        discovery_generation: DiscoveryGeneration,
        failed_run_id: Option<&StructuredRunId>,
        cx: &mut Context<Self>,
    ) -> Task<Result<JvmRemoteTestActionPlan>> {
        self.plan_remote_action_with_limit(node_id, discovery_generation, failed_run_id, None, cx)
    }

    pub fn plan_remote_action_with_limit(
        &self,
        node_id: &StructuredNodeId,
        discovery_generation: DiscoveryGeneration,
        failed_run_id: Option<&StructuredRunId>,
        report_maximum_bytes: Option<usize>,
        cx: &mut Context<Self>,
    ) -> Task<Result<JvmRemoteTestActionPlan>> {
        if let Some(maximum) = report_maximum_bytes {
            if !(1..=MAX_JUNIT_DISCOVERY_BYTES).contains(&maximum) {
                return Task::ready(Err(anyhow::anyhow!(
                    "Java report byte limit is unsupported"
                )));
            }
            if let Err(error) = self.ensure_bounded_remote_run_idle(cx) {
                return Task::ready(Err(error));
            }
        }
        if self.closed || self.disconnected {
            return Task::ready(Err(anyhow::anyhow!(
                "Reconnect to the project host to run Java tests"
            )));
        }
        let Some((project_id, client)) = self.remote.clone() else {
            return Task::ready(Err(anyhow::anyhow!(
                "Java test actions are already on the project host"
            )));
        };
        if self.current_remote_discovery != Some(discovery_generation) {
            return Task::ready(Err(anyhow::anyhow!(
                "The selected Java test discovery is stale; refresh Tests before running it"
            )));
        }
        let worktree_ids = self
            .worktree_store
            .read(cx)
            .visible_worktrees(cx)
            .map(|worktree| worktree.read(cx).id().to_proto())
            .take(MAX_REMOTE_JVM_WORKTREES)
            .collect::<Vec<_>>();
        let node_id = node_id.clone();
        let failed_run_id = failed_run_id.cloned();
        let client_generation = self.generation;
        let requested_run_id =
            report_maximum_bytes.map(|_| format!("jvm-bounded-{:032x}", rand::random::<u128>()));
        let cancellation_sender = self.remote_lifecycle_tx.clone();
        let action_protocol = if report_maximum_bytes.is_some() {
            JVM_TEST_BOUNDED_ACTION_PROTOCOL_VERSION
        } else {
            JVM_TEST_DISCOVERY_PROTOCOL_VERSION
        };
        cx.background_spawn(async move {
            let cancellation = match &requested_run_id {
                Some(run_id) => Some(std::sync::Arc::new(BoundedRemotePlanCancellation {request: proto::UpdateJvmTestRun {project_id, protocol_version: JVM_TEST_DISCOVERY_PROTOCOL_VERSION, discovery_generation: discovery_generation.0, run_id: run_id.clone(), node_id: node_id.0.clone(), task_id: format!("jvm-abandoned-{run_id}"), state: proto::JvmTestTaskState::Cancelled as i32, terminal_id: None, exit_code: None, success: false, termination_confirmed: true, diagnostic: None, worktree_ids: worktree_ids.clone(), abandon_unlaunched: true}, sender: cancellation_sender.context("Bounded Java plan cancellation channel unavailable")?, armed: std::sync::atomic::AtomicBool::new(true)})),
                None => None,
            };
            let response = client
                .request(proto::ResolveJvmTestAction {
                    project_id,
                    protocol_version: action_protocol,
                    discovery_generation: discovery_generation.0,
                    node_id: node_id.0.clone(),
                    failed_run_id: failed_run_id.map(|run_id| run_id.0),
                    worktree_ids: worktree_ids.clone(),
                    report_maximum_bytes: report_maximum_bytes.map(|maximum| maximum as u64),
                    requested_run_id: requested_run_id.clone(),
                })
                .await
                .context(
                    "The project host could not authorize this Java test. Save its source and refresh Tests; verify the host worktree is trusted. If Java runnable support is missing, reconnect to sync the installed Java extension and check the host log.",
                )?;
            validate_jvm_action_protocol(response.protocol_version, report_maximum_bytes.is_some())?;
            validate_remote_report_limit(report_maximum_bytes, response.report_maximum_bytes)?;
            ensure!(requested_run_id.as_ref().is_none_or(|run_id| run_id == &response.run_id), "The project host changed the private bounded Java reservation");
            ensure!(
                response.discovery_generation == discovery_generation.0
                    && response.node_id == node_id.0,
                "The remote Java test action changed during authorization"
            );
            validate_remote_jvm_field("Java test run ID", &response.run_id)?;
            ensure!(
                worktree_ids.contains(&response.worktree_id),
                "The remote Java test action is outside the visible worktree scope"
            );
            ensure!(
                response.template_json.len() <= MAX_REMOTE_JVM_PLAN_BYTES,
                "The remote Java test action plan exceeds the supported limit"
            );
            let template = serde_json::from_str::<TaskTemplate>(&response.template_json)
                .context("The remote Java test action plan is malformed")?;
            validate_remote_jvm_template(&template)?;
            Ok(JvmRemoteTestActionPlan {
                run_id: StructuredRunId(response.run_id),
                worktree_id: WorktreeId::from_proto(response.worktree_id),
                discovery_generation,
                node_id,
                template,
                client_generation,
                report_maximum_bytes,
                cancellation,
            })
        })
    }

    pub fn register_remote_run_handle(
        &mut self,
        plan: JvmRemoteTestActionPlan,
        handle: StructuredTaskHandle,
        cx: &mut Context<Self>,
    ) -> Result<()> {
        if plan.report_maximum_bytes.is_some() {
            self.ensure_bounded_remote_run_idle(cx)?;
        }
        let (project_id, _) = self
            .remote
            .as_ref()
            .context("The Java test action is not on a remote project")?;
        ensure!(
            !self.closed && !self.disconnected && self.generation == plan.client_generation,
            "The selected Java test changed before its Task started"
        );
        let provider = self
            .structured_store
            .read(cx)
            .state()
            .provider(&StructuredProviderId(JVM_JUNIT_PROVIDER_ID.to_string()))
            .context("Remote Java test discovery is unavailable")?;
        ensure!(
            provider.discovery_generation == plan.discovery_generation
                && provider
                    .nodes
                    .iter()
                    .any(|node| node.id == plan.node_id && node.kind == StructuredNodeKind::Case),
            "The selected Java test is no longer in the current discovery"
        );
        let worktree_ids = self
            .worktree_store
            .read(cx)
            .visible_worktrees(cx)
            .map(|worktree| worktree.read(cx).id().to_proto())
            .take(MAX_REMOTE_JVM_WORKTREES)
            .collect::<Vec<_>>();
        ensure!(
            worktree_ids.contains(&plan.worktree_id.to_proto()),
            "The selected Java test worktree is no longer visible"
        );
        let lifecycle_tx = self
            .remote_lifecycle_tx
            .clone()
            .context("The remote Java test lifecycle channel is unavailable")?;
        let project_id = *project_id;
        let run_id = plan.run_id.clone();
        for previous_handle in self.run_handles.values() {
            if !previous_handle.state().is_terminal() {
                previous_handle.cancel(cx);
            }
        }
        lifecycle_tx
            .unbounded_send(remote_jvm_lifecycle_to_proto(
                project_id,
                &plan,
                &StructuredTaskLifecycleEvent {
                    task_id: handle.task_id(),
                    state: StructuredTaskState::Queued,
                },
                worktree_ids.clone(),
            ))
            .context("The remote Java test lifecycle channel has closed")?;
        if let Some(cancellation) = &plan.cancellation {
            cancellation
                .armed
                .store(false, std::sync::atomic::Ordering::Relaxed);
        }
        handle.subscribe(cx, move |event, _| {
            let request =
                remote_jvm_lifecycle_to_proto(project_id, &plan, event, worktree_ids.clone());
            if let Err(error) = lifecycle_tx.unbounded_send(request) {
                log::warn!("Failed to queue remote Java test lifecycle: {error}");
            }
        });
        if self.run_handles.len() >= MAX_REMOTE_JVM_RUNS {
            self.run_handles
                .retain(|_, previous_handle| !previous_handle.state().is_terminal());
        }
        self.run_handles.insert(run_id, handle);
        Ok(())
    }

    pub fn validate_bounded_remote_plan(
        &self,
        plan: &JvmRemoteTestActionPlan,
        maximum_bytes: usize,
        cx: &gpui::App,
    ) -> Result<()> {
        ensure!(
            self.remote.is_some()
                && !self.closed
                && !self.disconnected
                && self.generation == plan.client_generation
                && self.current_remote_discovery == Some(plan.discovery_generation)
                && plan.report_maximum_bytes == Some(maximum_bytes),
            "Bounded remote Java test plan changed or was not acknowledged"
        );
        let provider = self
            .structured_store
            .read(cx)
            .state()
            .provider(&StructuredProviderId(JVM_JUNIT_PROVIDER_ID.into()))
            .context("Remote Java discovery is unavailable")?;
        ensure!(
            provider.discovery_generation == plan.discovery_generation
                && provider
                    .nodes
                    .iter()
                    .any(|node| node.id == plan.node_id && node.kind == StructuredNodeKind::Case)
                && self
                    .worktree_store
                    .read(cx)
                    .visible_worktrees(cx)
                    .any(|worktree| worktree.read(cx).id() == plan.worktree_id),
            "Selected remote Java test scope changed"
        );
        self.ensure_bounded_remote_run_idle(cx)
    }

    pub fn abandon_bounded_remote_plan(
        &self,
        plan: &JvmRemoteTestActionPlan,
        cx: &gpui::App,
    ) -> Result<()> {
        ensure!(
            plan.report_maximum_bytes.is_some(),
            "Only a bounded unlaunched plan can be abandoned"
        );
        let (project_id, _) = self
            .remote
            .as_ref()
            .context("Remote Java owner unavailable")?;
        ensure!(
            !self.closed && !self.disconnected && self.generation == plan.client_generation,
            "Remote Java connection changed; no plan replay"
        );
        let lifecycle = self
            .remote_lifecycle_tx
            .as_ref()
            .context("Remote Java lifecycle unavailable")?;
        let mut request = remote_jvm_lifecycle_to_proto(
            *project_id,
            plan,
            &StructuredTaskLifecycleEvent {
                task_id: TaskId(format!("jvm-abandoned-{}", plan.run_id.0)),
                state: StructuredTaskState::Cancelled {
                    terminal_id: None,
                    termination_confirmed: true,
                },
            },
            self.worktree_store
                .read(cx)
                .visible_worktrees(cx)
                .map(|worktree| worktree.read(cx).id().to_proto())
                .take(MAX_REMOTE_JVM_WORKTREES)
                .collect(),
        );
        request.abandon_unlaunched = true;
        lifecycle
            .unbounded_send(request)
            .context("Remote Java lifecycle closed")?;
        if let Some(cancellation) = &plan.cancellation {
            cancellation
                .armed
                .store(false, std::sync::atomic::Ordering::Relaxed);
        }
        Ok(())
    }

    pub fn fail_remote_action_plan(
        &self,
        plan: &JvmRemoteTestActionPlan,
        cx: &gpui::App,
    ) -> Result<()> {
        let (project_id, _) = self
            .remote
            .as_ref()
            .context("The Java test action is not on a remote project")?;
        ensure!(
            !self.closed && !self.disconnected && self.generation == plan.client_generation,
            "The remote Java test action is no longer current"
        );
        let worktree_ids = self
            .worktree_store
            .read(cx)
            .visible_worktrees(cx)
            .map(|worktree| worktree.read(cx).id().to_proto())
            .take(MAX_REMOTE_JVM_WORKTREES)
            .collect::<Vec<_>>();
        ensure!(
            worktree_ids.contains(&plan.worktree_id.to_proto()),
            "The remote Java test worktree is no longer visible"
        );
        let lifecycle_tx = self
            .remote_lifecycle_tx
            .as_ref()
            .context("The remote Java test lifecycle channel is unavailable")?;
        let task_id = TaskId(format!("jvm-action-{}", plan.run_id.0));
        for state in [
            StructuredTaskState::Queued,
            StructuredTaskState::SpawnError {
                message:
                    "The selected Java test Task could not be prepared; inspect the workspace error"
                        .to_string(),
            },
        ] {
            lifecycle_tx
                .unbounded_send(remote_jvm_lifecycle_to_proto(
                    *project_id,
                    plan,
                    &StructuredTaskLifecycleEvent {
                        task_id: task_id.clone(),
                        state,
                    },
                    worktree_ids.clone(),
                ))
                .context("The remote Java test lifecycle channel has closed")?;
        }
        Ok(())
    }

    #[cfg(any(test, feature = "test-support"))]
    pub fn gate_next_discovery_reply_for_test(
        &mut self,
    ) -> (
        futures::channel::oneshot::Receiver<()>,
        futures::channel::oneshot::Sender<()>,
        futures::channel::oneshot::Receiver<()>,
    ) {
        let (started_sender, started_receiver) = futures::channel::oneshot::channel();
        let (release_sender, release_receiver) = futures::channel::oneshot::channel();
        let (completed_sender, completed_receiver) = futures::channel::oneshot::channel();
        self.discovery_reply_gate = Some((started_sender, release_receiver, completed_sender));
        (started_receiver, release_sender, completed_receiver)
    }

    #[cfg(any(test, feature = "test-support"))]
    pub fn gate_next_action_reply_for_test(
        &mut self,
    ) -> (
        futures::channel::oneshot::Receiver<()>,
        futures::channel::oneshot::Sender<()>,
    ) {
        let (started_sender, started_receiver) = futures::channel::oneshot::channel();
        let (release_sender, release_receiver) = futures::channel::oneshot::channel();
        self.action_reply_gate = Some((started_sender, release_receiver));
        (started_receiver, release_sender)
    }

    pub fn plan_selected_case(
        &self,
        node_id: &StructuredNodeId,
        discovery_generation: DiscoveryGeneration,
        cx: &gpui::App,
    ) -> Result<JvmSelectedTestPlan> {
        ensure!(!self.closed, "The Java test project has closed");
        ensure!(
            self.remote.is_none(),
            "Remote Java test actions need the host action protocol"
        );
        ensure!(
            self.generation == discovery_generation.0,
            "The selected Java test belongs to a stale discovery generation"
        );
        let provider = self
            .structured_store
            .read(cx)
            .state()
            .provider(&StructuredProviderId(JVM_JUNIT_PROVIDER_ID.to_string()))
            .context("Java test discovery is unavailable")?;
        ensure!(
            provider.discovery_generation == discovery_generation
                && provider
                    .nodes
                    .iter()
                    .any(|node| &node.id == node_id && node.kind == StructuredNodeKind::Case),
            "The selected Java test is no longer in the current discovery"
        );
        let selector = self.selectors.get(node_id).context(
            "This Java test has no verified method selector; save its source and refresh Tests",
        )?;
        let build_tool = match selector.scope_path.path.file_name() {
            Some("pom.xml") => JvmTestBuildTool::Maven,
            Some("build.gradle" | "build.gradle.kts") => JvmTestBuildTool::Gradle,
            _ => bail!("The selected Java test has no supported module manifest"),
        };
        let worktree = self
            .worktree_store
            .read(cx)
            .visible_worktrees(cx)
            .find(|worktree| worktree.read(cx).id() == selector.scope_path.worktree_id)
            .context("The selected Java test worktree is no longer visible")?;
        let worktree = worktree.read(cx);
        let snapshot = worktree.snapshot();
        let manifest = snapshot
            .entry_for_path(selector.scope_path.path.as_ref())
            .context("The selected Java test module manifest is no longer visible")?;
        ensure!(
            jvm_build_entry_is_available(manifest),
            "The selected Java test module manifest is unavailable"
        );
        let wrapper_name = match (build_tool, worktree.path_style()) {
            (JvmTestBuildTool::Maven, PathStyle::Unix) => "mvnw",
            (JvmTestBuildTool::Maven, PathStyle::Windows) => "mvnw.cmd",
            (JvmTestBuildTool::Gradle, PathStyle::Unix) => "gradlew",
            (JvmTestBuildTool::Gradle, PathStyle::Windows) => "gradlew.bat",
        };
        let wrapper_path = RelPath::from_unix_str(wrapper_name)?;
        let wrapper = snapshot
            .entry_for_path(wrapper_path)
            .context("The project build wrapper is missing; add the Maven or Gradle wrapper to run this Java test")?;
        ensure!(
            jvm_build_entry_is_available(wrapper),
            "The project build wrapper is unavailable"
        );
        Ok(JvmSelectedTestPlan {
            worktree_id: selector.scope_path.worktree_id,
            discovery_generation,
            node_id: node_id.clone(),
            selector: selector.clone(),
            build_tool,
            template: selected_junit_task(selector, build_tool, worktree.path_style())?,
        })
    }

    pub fn plan_failed_case(
        &self,
        node_id: &StructuredNodeId,
        discovery_generation: DiscoveryGeneration,
        failed_run_id: &StructuredRunId,
        cx: &gpui::App,
    ) -> Result<JvmSelectedTestPlan> {
        let plan = self.plan_selected_case(node_id, discovery_generation, cx)?;
        let provider_id = StructuredProviderId(JVM_JUNIT_PROVIDER_ID.to_string());
        let provider = self
            .structured_store
            .read(cx)
            .state()
            .provider(&provider_id)
            .context("Java test discovery is unavailable")?;
        let run = provider
            .current_run
            .as_ref()
            .or(provider.last_complete_run.as_ref())
            .context("The selected Java test has no current result")?;
        ensure!(
            run.id == *failed_run_id
                && run.discovery_generation == discovery_generation
                && run.phase == StructuredRunPhase::Completed
                && run.scope_node_ids.contains(node_id)
                && run
                    .events
                    .iter()
                    .rev()
                    .find(|event| &event.node_id == node_id)
                    .is_some_and(|event| event.state == StructuredNodeState::Failed),
            "The selected Java test is no longer a current failed case"
        );
        Ok(plan)
    }

    pub fn capture_report_version(
        &self,
        plan: &JvmSelectedTestPlan,
        cx: &mut Context<Self>,
    ) -> Task<Result<JunitReportVersion>> {
        self.capture_report_version_with_limit(plan, MAX_JUNIT_DISCOVERY_BYTES, cx)
    }

    pub fn capture_report_version_with_limit(
        &self,
        plan: &JvmSelectedTestPlan,
        maximum_bytes: usize,
        cx: &mut Context<Self>,
    ) -> Task<Result<JunitReportVersion>> {
        let prepared = (|| {
            ensure!(
                (1..=MAX_JUNIT_DISCOVERY_BYTES).contains(&maximum_bytes),
                "Invalid selected JUnit evidence byte limit"
            );
            ensure!(
                self.plan_selected_case(&plan.node_id, plan.discovery_generation, cx)? == *plan,
                "The selected Java test changed before its report was captured"
            );
            let fs = self
                .worktree_store
                .read(cx)
                .fs()
                .context("The Java test project host has no local filesystem")?;
            let worktree = self
                .worktree_store
                .read(cx)
                .visible_worktrees(cx)
                .find(|worktree| worktree.read(cx).id() == plan.worktree_id)
                .context("The selected Java test worktree is no longer visible")?;
            let root = worktree.read(cx).abs_path();
            Ok((fs, root))
        })();
        let (fs, root) = match prepared {
            Ok(value) => value,
            Err(error) => return Task::ready(Err(error)),
        };
        let plan = plan.clone();
        cx.background_spawn(async move {
            let reports = collect_junit_reports_with_budget(
                fs.as_ref(),
                root.as_ref(),
                plan.selector.scope_path.clone(),
                &plan.selector.module_path,
                plan.build_tool,
                maximum_bytes,
            )
            .await?
            .0;
            let (_, version) = matching_junit_case(&reports, &plan.selector)?;
            Ok(version.clone())
        })
    }

    pub fn register_run_handle(
        &mut self,
        plan: JvmSelectedTestPlan,
        previous_report: JunitReportVersion,
        handle: StructuredTaskHandle,
        cx: &mut Context<Self>,
    ) -> Result<StructuredRunId> {
        self.register_run_handle_with_limit(
            plan,
            previous_report,
            handle,
            MAX_JUNIT_DISCOVERY_BYTES,
            cx,
        )
    }

    pub fn register_run_handle_with_limit(
        &mut self,
        plan: JvmSelectedTestPlan,
        previous_report: JunitReportVersion,
        handle: StructuredTaskHandle,
        maximum_bytes: usize,
        cx: &mut Context<Self>,
    ) -> Result<StructuredRunId> {
        ensure!(
            (1..=MAX_JUNIT_DISCOVERY_BYTES).contains(&maximum_bytes),
            "Invalid selected JUnit evidence byte limit"
        );
        ensure!(
            self.plan_selected_case(&plan.node_id, plan.discovery_generation, cx)? == plan,
            "The selected Java test changed before its Task started"
        );
        self.next_run_id = self.next_run_id.wrapping_add(1);
        let run_id = StructuredRunId(format!("jvm-junit-run-{}", self.next_run_id));
        let provider_id = StructuredProviderId(JVM_JUNIT_PROVIDER_ID.to_string());
        self.structured_store.update(cx, |store, cx| {
            let project_generation = store.state().project_generation();
            store.state_mut().begin_run(
                project_generation,
                &provider_id,
                StructuredRun::new(
                    run_id.clone(),
                    plan.discovery_generation,
                    vec![plan.node_id.clone()],
                ),
            )?;
            cx.emit(StructuredExecutionStoreEvent::Changed(provider_id.clone()));
            anyhow::Ok(())
        })?;
        for previous_handle in self.run_handles.values() {
            if !previous_handle.state().is_terminal() {
                previous_handle.cancel(cx);
            }
        }
        if self.run_handles.len() >= 32 {
            self.run_handles
                .retain(|_, previous_handle| !previous_handle.state().is_terminal());
        }
        self.run_handles.insert(run_id.clone(), handle.clone());
        let provider = cx.entity().downgrade();
        let observed_run_id = run_id.clone();
        handle.subscribe(cx, move |event, cx| {
            let event = event.clone();
            let provider = provider.clone();
            let run_id = observed_run_id.clone();
            let plan = plan.clone();
            let previous_report = previous_report.clone();
            cx.defer(move |cx| {
                let result = provider.update(cx, |provider, cx| {
                    provider.handle_run_event(
                        &run_id,
                        &plan,
                        &previous_report,
                        &event,
                        maximum_bytes,
                        cx,
                    )
                });
                match result {
                    Ok(Ok(())) => {}
                    Ok(Err(error)) => log::warn!("Rejected Java test Task event: {error:#}"),
                    Err(error) => log::debug!("Java test project was dropped: {error:#}"),
                }
            });
        });
        Ok(run_id)
    }

    pub fn has_active_runs(&self) -> bool {
        self.run_handles
            .values()
            .any(|handle| !handle.state().is_terminal())
    }

    pub fn cancel_run(&self, run_id: &StructuredRunId, cx: &mut gpui::App) -> Result<()> {
        let handle = self
            .run_handles
            .get(run_id)
            .context("The selected Java test run is no longer active")?;
        ensure!(
            handle.cancel(cx),
            "The selected Java test run is already complete"
        );
        Ok(())
    }

    pub fn reveal_run_terminal(
        &self,
        run_id: &StructuredRunId,
        window: &mut gpui::Window,
        cx: &mut gpui::App,
    ) -> Result<()> {
        let handle = self
            .run_handles
            .get(run_id)
            .context("The selected Java test run has no retained Task terminal")?;
        ensure!(
            handle.reveal_terminal(window, cx),
            "The selected Java test run has no available Task terminal"
        );
        Ok(())
    }

    fn handle_run_event(
        &mut self,
        run_id: &StructuredRunId,
        plan: &JvmSelectedTestPlan,
        previous_report: &JunitReportVersion,
        event: &StructuredTaskLifecycleEvent,
        maximum_bytes: usize,
        cx: &mut Context<Self>,
    ) -> Result<()> {
        if self.closed
            || self.generation != plan.discovery_generation.0
            || !self.run_handles.contains_key(run_id)
        {
            log::debug!("Ignored an obsolete Java test Task event");
            return Ok(());
        }
        self.process_run_event(run_id, plan, previous_report, event, maximum_bytes, cx)
    }

    fn process_run_event(
        &mut self,
        run_id: &StructuredRunId,
        plan: &JvmSelectedTestPlan,
        previous_report: &JunitReportVersion,
        event: &StructuredTaskLifecycleEvent,
        maximum_bytes: usize,
        cx: &mut Context<Self>,
    ) -> Result<()> {
        let provider_id = StructuredProviderId(JVM_JUNIT_PROVIDER_ID.to_string());
        match &event.state {
            StructuredTaskState::Queued => Ok(()),
            StructuredTaskState::Running { .. } => self.structured_store.update(cx, |store, cx| {
                let project_generation = store.state().project_generation();
                store.state_mut().set_run_phase(
                    project_generation,
                    &provider_id,
                    run_id,
                    StructuredRunPhase::Running,
                    None,
                )?;
                cx.emit(StructuredExecutionStoreEvent::Changed(provider_id));
                anyhow::Ok(())
            }),
            StructuredTaskState::Cancelled { .. } | StructuredTaskState::SpawnError { .. } => {
                self.structured_store.update(cx, |store, cx| {
                    store.apply_task_lifecycle(&provider_id, run_id, event, cx)?;
                    anyhow::Ok(())
                })
            }
            StructuredTaskState::Completed {
                success, exit_code, ..
            } => {
                let fs = self
                    .worktree_store
                    .read(cx)
                    .fs()
                    .context("The Java test host filesystem is unavailable")?;
                let worktree = self
                    .worktree_store
                    .read(cx)
                    .visible_worktrees(cx)
                    .find(|worktree| worktree.read(cx).id() == plan.worktree_id)
                    .context("The Java test worktree is no longer visible")?;
                let root = worktree.read(cx).abs_path();
                let plan = plan.clone();
                let previous_report = previous_report.clone();
                let run_id = run_id.clone();
                let success = *success;
                let exit_code = *exit_code;
                cx.spawn(async move |this, cx| {
                    let result = async {
                        let reports = collect_junit_reports_with_budget(
                            fs.as_ref(),
                            root.as_ref(),
                            plan.selector.scope_path.clone(),
                            &plan.selector.module_path,
                            plan.build_tool,
                            maximum_bytes,
                        )
                        .await?
                        .0;
                        Ok::<_, anyhow::Error>(
                            fresh_junit_case(&reports, &plan.selector, &previous_report)?.clone(),
                        )
                    }
                    .await;
                    this.update(cx, |provider, cx| {
                        provider.finish_run(&run_id, &plan, result, success, exit_code, cx)
                    })??;
                    anyhow::Ok(())
                })
                .detach_and_log_err(cx);
                Ok(())
            }
        }
    }

    fn finish_run(
        &mut self,
        run_id: &StructuredRunId,
        plan: &JvmSelectedTestPlan,
        result: Result<JunitCase>,
        task_success: bool,
        exit_code: Option<i32>,
        cx: &mut Context<Self>,
    ) -> Result<()> {
        let provider_id = StructuredProviderId(JVM_JUNIT_PROVIDER_ID.to_string());
        if self.closed
            || self.generation != plan.discovery_generation.0
            || !self
                .structured_store
                .read(cx)
                .state()
                .provider(&provider_id)
                .and_then(|snapshot| snapshot.current_run.as_ref())
                .is_some_and(|run| &run.id == run_id && !run.phase.is_terminal())
        {
            log::debug!("Ignored an obsolete Java test result");
            return Ok(());
        }
        self.structured_store.update(cx, |store, cx| {
            let project_generation = store.state().project_generation();
            let diagnostic = match result {
                Ok(case) => {
                    store.state_mut().apply_event(
                        project_generation,
                        &provider_id,
                        run_id,
                        StructuredExecutionEvent {
                            sequence: 0,
                            node_id: plan.node_id.clone(),
                            state: case.state,
                            duration_millis: None,
                            message: case.message,
                            location: None,
                        },
                        None,
                    )?;
                    (!task_success).then(|| match exit_code {
                        Some(exit_code) => format!(
                            "The Task exited with status {exit_code} after writing the selected JUnit result"
                        ),
                        None => "The Task failed after writing the selected JUnit result".to_string(),
                    })
                }
                Err(error) => {
                    log::warn!("Could not verify a fresh selected JUnit result: {error:#}");
                    Some(
                        "The Task ended without a fresh readable JUnit result for the selected test; inspect its terminal and refresh Tests"
                            .to_string(),
                    )
                }
            };
            store.state_mut().set_run_phase(
                project_generation,
                &provider_id,
                run_id,
                StructuredRunPhase::Completed,
                diagnostic.as_deref(),
            )?;
            cx.emit(StructuredExecutionStoreEvent::Changed(provider_id));
            anyhow::Ok(())
        })
    }

    fn subscribe_to_worktree_restrictions(cx: &mut Context<Self>) -> Option<Subscription> {
        let trusted_worktrees = TrustedWorktrees::try_get_global(cx)?;
        Some(cx.subscribe(&trusted_worktrees, |store, _, event, cx| {
            if matches!(event, TrustedWorktreesEvent::Restricted(worktree_store, _) if *worktree_store == store.worktree_store.downgrade()) {
                store.invalidate(cx);
            }
        }))
    }

    pub fn local(
        worktree_store: Entity<WorktreeStore>,
        buffer_store: Entity<BufferStore>,
        structured_store: Entity<StructuredExecutionStore>,
        cx: &mut Context<Self>,
    ) -> Self {
        let worktree_subscription = cx.subscribe(&worktree_store, |store, _, event, cx| {
            let selection_changed = match event {
                WorktreeStoreEvent::WorktreeAdded(_)
                | WorktreeStoreEvent::WorktreeRemoved(_, _)
                | WorktreeStoreEvent::WorktreeReleased(_, _)
                | WorktreeStoreEvent::WorktreeDeletedEntry(_, _) => true,
                WorktreeStoreEvent::WorktreeUpdatedEntries(_, changes) => changes
                    .iter()
                    .any(|(path, _, _)| affects_jvm_test_selection(path)),
                _ => false,
            };
            if selection_changed {
                store.invalidate(cx);
            }
        });
        Self {
            remote: None,
            worktree_store,
            buffer_store: buffer_store.downgrade(),
            structured_store,
            generation: 0,
            next_run_id: 0,
            closed: false,
            disconnected: false,
            selectors: HashMap::new(),
            run_handles: HashMap::new(),
            current_remote_discovery: None,
            active_remote_request_id: None,
            active_remote_request_abort: None,
            active_remote_requests: HashMap::new(),
            cancelled_remote_requests: HashSet::new(),
            authorized_remote_runs: HashMap::new(),
            remote_lifecycle_tx: None,
            _remote_lifecycle_task: Task::ready(()),
            #[cfg(any(test, feature = "test-support"))]
            discovery_reply_gate: None,
            #[cfg(any(test, feature = "test-support"))]
            action_reply_gate: None,
            _subscriptions: std::iter::once(worktree_subscription)
                .chain(Self::subscribe_to_worktree_restrictions(cx))
                .collect(),
        }
    }

    pub fn remote(
        project_id: u64,
        client: AnyProtoClient,
        worktree_store: Entity<WorktreeStore>,
        buffer_store: Entity<BufferStore>,
        structured_store: Entity<StructuredExecutionStore>,
        cx: &mut Context<Self>,
    ) -> Self {
        let worktree_subscription = cx.subscribe(&worktree_store, |store, _, event, cx| {
            if matches!(
                event,
                WorktreeStoreEvent::WorktreeAdded(_)
                    | WorktreeStoreEvent::WorktreeRemoved(_, _)
                    | WorktreeStoreEvent::WorktreeReleased(_, _)
            ) {
                store.invalidate(cx);
            }
        });
        let (remote_lifecycle_tx, mut remote_lifecycle_rx) = mpsc::unbounded();
        let lifecycle_client = client.clone();
        let structured_results = structured_store.clone();
        let remote_lifecycle_task = cx.spawn(async move |_, cx| {
            while let Some(request) = remote_lifecycle_rx.next().await {
                let terminal = match remote_jvm_lifecycle_from_proto(&request) {
                    Ok(event) => event.state.is_terminal(),
                    Err(error) => {
                        log::warn!("Invalid remote Java test lifecycle: {error:#}");
                        continue;
                    }
                };
                let run_id = StructuredRunId(request.run_id.clone());
                let abandon_unlaunched = request.abandon_unlaunched;
                match lifecycle_client.request(request).await {
                    Ok(_) => {}
                    Err(error) => {
                        log::warn!("Failed to publish remote Java test lifecycle: {error:#}");
                        continue;
                    }
                }
                if abandon_unlaunched {
                    continue;
                }
                let provider_id = StructuredProviderId(JVM_JUNIT_PROVIDER_ID.to_string());
                for attempt in 0..10 {
                    let refresh = structured_results.update(cx, |store, cx| {
                        store.refresh_provider(provider_id.clone(), cx)
                    });
                    match refresh.await {
                        Ok(snapshot)
                            if snapshot.current_run.as_ref().is_some_and(|run| {
                                run.id == run_id && (!terminal || run.phase.is_terminal())
                            }) =>
                        {
                            break;
                        }
                        Ok(_) if attempt < 9 => {
                            cx.background_executor()
                                .timer(Duration::from_millis(100))
                                .await;
                        }
                        Ok(_) => {
                            log::warn!("The remote Java test run did not reach the Tests view");
                        }
                        Err(error) => {
                            log::warn!("Failed to refresh the remote Java test run: {error:#}");
                            break;
                        }
                    }
                }
            }
        });
        Self {
            remote: Some((project_id, client)),
            worktree_store,
            buffer_store: buffer_store.downgrade(),
            structured_store,
            generation: 0,
            next_run_id: 0,
            closed: false,
            disconnected: false,
            selectors: HashMap::new(),
            run_handles: HashMap::new(),
            current_remote_discovery: None,
            active_remote_request_id: None,
            active_remote_request_abort: None,
            active_remote_requests: HashMap::new(),
            cancelled_remote_requests: HashSet::new(),
            authorized_remote_runs: HashMap::new(),
            remote_lifecycle_tx: Some(remote_lifecycle_tx),
            _remote_lifecycle_task: remote_lifecycle_task,
            #[cfg(any(test, feature = "test-support"))]
            discovery_reply_gate: None,
            #[cfg(any(test, feature = "test-support"))]
            action_reply_gate: None,
            _subscriptions: std::iter::once(worktree_subscription)
                .chain(Self::subscribe_to_worktree_restrictions(cx))
                .collect(),
        }
    }

    pub fn close(&mut self, cx: &mut Context<Self>) {
        self.closed = true;
        self.invalidate(cx);
        for (_, abort) in self.active_remote_requests.drain() {
            abort.abort();
        }
        self.cancelled_remote_requests.clear();
        self.authorized_remote_runs.clear();
        self.run_handles.clear();
    }

    pub fn disconnected_from_host(&mut self, cx: &mut Context<Self>) {
        if self.remote.is_some() {
            self.disconnected = true;
            self.invalidate(cx);
        }
    }

    pub fn reconnected_to_host(&mut self, cx: &mut Context<Self>) {
        if self.remote.is_some() && self.disconnected {
            self.disconnected = false;
            self.invalidate(cx);
        }
    }

    pub fn invalidate(&mut self, cx: &mut Context<Self>) {
        if let Some((project_id, client)) = &self.remote {
            for handle in self.run_handles.values() {
                if !handle.state().is_terminal() {
                    handle.cancel(cx);
                }
            }
            if let Some(request_id) = self.active_remote_request_id.take() {
                let project_id = *project_id;
                let client = client.clone();
                cx.background_spawn(async move {
                    client
                        .request(proto::CancelJvmTestDiscovery {
                            project_id,
                            request_id,
                        })
                        .await?;
                    anyhow::Ok(())
                })
                .detach_and_log_err(cx);
            }
            if let Some(abort) = self.active_remote_request_abort.take() {
                abort.abort();
            }
            self.generation = self.generation.wrapping_add(1);
            self.current_remote_discovery = None;
            self.selectors.clear();
            return;
        }
        for handle in self.run_handles.values() {
            if !handle.state().is_terminal() {
                handle.cancel(cx);
            }
        }
        let provider_id = StructuredProviderId(JVM_JUNIT_PROVIDER_ID.to_string());
        let result = self.structured_store.update(cx, |store, cx| {
            let current = store
                .state()
                .provider(&provider_id)
                .and_then(|snapshot| snapshot.current_run.clone());
            if let Some(run) = current.filter(|run| !run.phase.is_terminal()) {
                let project_generation = store.state().project_generation();
                if let Some(node_id) = run.scope_node_ids.first()
                    && run.scope_node_ids.len() == 1
                {
                    store.state_mut().apply_event(
                        project_generation,
                        &provider_id,
                        &run.id,
                        StructuredExecutionEvent {
                            sequence: run.next_sequence,
                            node_id: node_id.clone(),
                            state: StructuredNodeState::Cancelled,
                            duration_millis: None,
                            message: Some(
                                "Java test selection changed during the Task".to_string(),
                            ),
                            location: None,
                        },
                        None,
                    )?;
                }
                store.state_mut().set_run_phase(
                    project_generation,
                    &provider_id,
                    &run.id,
                    StructuredRunPhase::Cancelled,
                    Some("Java test selection changed; refresh Tests before running again"),
                )?;
                cx.emit(StructuredExecutionStoreEvent::Changed(provider_id.clone()));
            }
            anyhow::Ok(())
        });
        if let Err(error) = result {
            log::warn!("Could not invalidate an active Java test run: {error:#}");
        }
        self.generation = self.generation.wrapping_add(1);
        self.selectors.clear();
        self.authorized_remote_runs.clear();
    }

    pub fn refresh(&mut self, cx: &mut Context<Self>) -> Task<Result<StructuredProviderSnapshot>> {
        self.refresh_scoped(None, cx)
    }

    fn refresh_scoped(
        &mut self,
        allowed_worktrees: Option<HashSet<WorktreeId>>,
        cx: &mut Context<Self>,
    ) -> Task<Result<StructuredProviderSnapshot>> {
        if self.closed {
            return Task::ready(Err(anyhow::anyhow!("The Java test project has closed")));
        }
        if self.disconnected {
            return Task::ready(Err(anyhow::anyhow!(
                "Reconnect to the project host to load Java tests"
            )));
        }
        if let Some((project_id, client)) = self.remote.clone() {
            if self
                .run_handles
                .values()
                .any(|handle| !handle.state().is_terminal())
            {
                return Task::ready(Err(anyhow::anyhow!(
                    "Wait for the selected Java test Task to finish before refreshing Tests"
                )));
            }
            self.invalidate(cx);
            let generation = self.generation;
            let structured_store = self.structured_store.clone();
            let worktree_ids = self
                .worktree_store
                .read(cx)
                .visible_worktrees(cx)
                .map(|worktree| worktree.read(cx).id().to_proto())
                .collect();
            let (abort_handle, abort_registration) = AbortHandle::new_pair();
            self.active_remote_request_id = Some(generation);
            self.active_remote_request_abort = Some(abort_handle);
            return cx.spawn(async move |this, cx| {
                let operation = async {
                    let response = client
                        .request(proto::GetJvmTestDiscovery {
                            project_id,
                            protocol_version: JVM_TEST_DISCOVERY_PROTOCOL_VERSION,
                            worktree_ids,
                            request_id: generation,
                        })
                        .await
                        .map_err(|error| {
                            log::warn!("Remote Java test discovery failed: {error:#}");
                            anyhow::anyhow!(
                                "The project host does not support Java test discovery or is disconnected"
                            )
                        })?;
                    ensure!(
                        response.protocol_version == JVM_TEST_DISCOVERY_PROTOCOL_VERSION
                            && response.request_id == generation,
                        "Java test discovery protocol mismatch"
                    );
                    ensure!(
                        response.status != proto::StructuredProviderStatus::Mismatch as i32,
                        "Java test discovery protocol mismatch"
                    );
                    let snapshot = structured_store
                        .update(cx, |store, cx| {
                            store.refresh_provider(
                                StructuredProviderId(JVM_JUNIT_PROVIDER_ID.to_string()),
                                cx,
                            )
                        })
                        .await?;
                    ensure!(
                        snapshot.discovery_generation.0 == response.discovery_generation,
                        "Java test discovery changed during remote refresh"
                    );
                    Ok(snapshot)
                };
                let result = Abortable::new(operation, abort_registration)
                    .await
                    .map_err(|_| anyhow::anyhow!("Java test discovery was cancelled"))?;
                this.update(cx, |store, _| {
                    if store.active_remote_request_id == Some(generation) {
                        store.active_remote_request_id = None;
                        store.active_remote_request_abort = None;
                    }
                    ensure!(store.generation == generation && !store.closed, "Stale Java test discovery");
                    let snapshot = result?;
                    store.current_remote_discovery = Some(snapshot.discovery_generation);
                    Ok(snapshot)
                })?
            });
        }
        let active_run = self
            .structured_store
            .read(cx)
            .state()
            .provider(&StructuredProviderId(JVM_JUNIT_PROVIDER_ID.to_string()))
            .and_then(|snapshot| snapshot.current_run.as_ref())
            .is_some_and(|run| !run.phase.is_terminal());
        if active_run {
            return Task::ready(Err(anyhow::anyhow!(
                "Wait for the selected Java test Task to finish before refreshing Tests"
            )));
        }
        self.invalidate(cx);
        let generation = DiscoveryGeneration(self.generation);
        let scopes = self.module_scopes(allowed_worktrees.as_ref(), cx);
        let fs = self.worktree_store.read(cx).fs();
        let structured_store = self.structured_store.clone();
        let buffer_store = self.buffer_store.upgrade();
        cx.spawn(async move |this, cx| {
            let fs = fs.context("The Java test project host has no local filesystem")?;
            let buffer_store = buffer_store.context("The Java test project buffer store has closed")?;
            let (scopes, visible_sources) = scopes?;
            let mut reports = Vec::new();
            let mut remaining_bytes = MAX_JUNIT_DISCOVERY_BYTES;
            let mut total_cases = 0usize;
            let mut failures = 0usize;
            for scope in scopes {
                match collect_junit_reports_with_budget(
                    fs.as_ref(),
                    &scope.root,
                    scope.manifest,
                    &scope.module_path,
                    scope.build_tool,
                    remaining_bytes,
                )
                .await
                {
                    Ok((mut module_reports, bytes_read)) => {
                        remaining_bytes -= bytes_read;
                        filter_visible_report_cases(&mut module_reports, &visible_sources);
                        let module_cases = module_reports
                            .iter()
                            .map(|report| report.cases.len())
                            .sum::<usize>();
                        if reports.len() + module_reports.len() > MAX_JUNIT_REPORT_FILES
                            || total_cases + module_cases > MAX_JUNIT_CASES
                        {
                            failures += 1;
                            break;
                        }
                        total_cases += module_cases;
                        reports.extend(module_reports);
                    }
                    Err(error) => {
                        failures += 1;
                        log::warn!("Could not collect a Java test module's reports: {error:#}");
                    }
                }
            }
            let (verified_cases, skipped_sources) =
                verify_report_sources(&reports, &visible_sources, &buffer_store, cx).await;
            let discovery = JvmTestDiscovery::from_reports(&reports, generation, &verified_cases)?;
            let mut snapshot = discovery.snapshot;
            let display_only_cases = snapshot
                .nodes
                .iter()
                .filter(|node| {
                    node.kind == StructuredNodeKind::Case
                        && !discovery.selectors.contains_key(&node.id)
                })
                .count();
            if !reports.is_empty() {
                snapshot.status = StructuredProviderStatus::Stale;
                snapshot.diagnostic = Some(
                    "Test names come from JUnit reports; only tests run in this session have verified results"
                        .to_string(),
                );
            }
            if failures > 0 {
                if reports.is_empty() {
                    snapshot.status = StructuredProviderStatus::Partial;
                }
                snapshot.partial = true;
                let failure_diagnostic = format!(
                    "Could not read reports for {failures} Java test modules on the project host"
                );
                snapshot.diagnostic = Some(match snapshot.diagnostic.take() {
                    Some(existing) => format!("{existing}; {failure_diagnostic}"),
                    None => failure_diagnostic,
                });
            }
            if skipped_sources > 0 {
                snapshot.partial = true;
                if reports.is_empty() {
                    snapshot.status = StructuredProviderStatus::Partial;
                }
                let source_diagnostic = format!(
                    "Could not verify {skipped_sources} Java test sources; save edited files or reduce their size, wait for Java syntax loading, then refresh Tests to make cases selectable"
                );
                snapshot.diagnostic = Some(match snapshot.diagnostic.take() {
                    Some(existing) => format!("{existing}; {source_diagnostic}"),
                    None => source_diagnostic,
                });
            }
            if display_only_cases > 0 {
                snapshot.partial = true;
                let source_diagnostic = format!(
                    "{display_only_cases} reported Java test cases are display-only because no unique @Test source method was verified; save source files and refresh Tests, then check the Java extension if this persists"
                );
                snapshot.diagnostic = Some(match snapshot.diagnostic.take() {
                    Some(existing) => format!("{existing}; {source_diagnostic}"),
                    None => source_diagnostic,
                });
            }
            this.update(cx, |store, _| {
                ensure!(
                    !store.closed && store.generation == generation.0,
                    "stale Java test discovery generation"
                );
                store.selectors = discovery.selectors;
                Ok(())
            })??;
            structured_store.update(cx, |store, cx| {
                let project_generation = store.state().project_generation();
                store
                    .state_mut()
                    .apply_discovery(project_generation, snapshot.clone(), None)?;
                cx.emit(StructuredExecutionStoreEvent::Changed(
                    StructuredProviderId(JVM_JUNIT_PROVIDER_ID.to_string()),
                ));
                anyhow::Ok(())
            })?;
            Ok(snapshot)
        })
    }

    fn module_scopes(
        &self,
        allowed_worktrees: Option<&HashSet<WorktreeId>>,
        cx: &gpui::App,
    ) -> Result<(Vec<JvmModuleScope>, HashMap<ProjectPath, u64>)> {
        let mut scopes = Vec::new();
        let mut seen = HashSet::new();
        let mut visible_sources = HashMap::new();
        for worktree in self.worktree_store.read(cx).visible_worktrees(cx) {
            let worktree = worktree.read(cx);
            let root = worktree.abs_path();
            let worktree_id = worktree.id();
            if allowed_worktrees.is_some_and(|allowed| !allowed.contains(&worktree_id)) {
                continue;
            }
            for entry in worktree.snapshot().files(false, 0) {
                if entry.is_private || entry.is_external || entry.is_fifo {
                    continue;
                }
                if entry.path.extension() == Some("java") {
                    visible_sources.insert(
                        ProjectPath {
                            worktree_id,
                            path: entry.path.clone(),
                        },
                        entry.size,
                    );
                }
                let build_tool = match entry.path.file_name() {
                    Some("pom.xml") => JvmTestBuildTool::Maven,
                    Some("build.gradle" | "build.gradle.kts") => JvmTestBuildTool::Gradle,
                    _ => continue,
                };
                let module = entry.path.parent().map_or("", |path| path.as_unix_str());
                let module_path = if module.is_empty() { "." } else { module }.to_string();
                if !seen.insert((worktree_id, module_path.clone(), build_tool)) {
                    continue;
                }
                scopes.push(JvmModuleScope {
                    root: root.clone(),
                    manifest: ProjectPath {
                        worktree_id,
                        path: entry.path.clone(),
                    },
                    module_path,
                    build_tool,
                });
                ensure!(
                    scopes.len() <= MAX_JUNIT_MODULES,
                    "Java test discovery has too many build modules"
                );
            }
        }
        Ok((scopes, visible_sources))
    }
}

fn affects_jvm_test_selection(path: &RelPath) -> bool {
    let path_text = path.as_unix_str();
    (path.extension() == Some("java")
        && (path_text.starts_with("src/test/java/") || path_text.contains("/src/test/java/")))
        || matches!(
            path.file_name(),
            Some(
                "pom.xml"
                    | "build.gradle"
                    | "build.gradle.kts"
                    | "mvnw"
                    | "mvnw.cmd"
                    | "gradlew"
                    | "gradlew.bat"
            )
        )
}

fn filter_visible_report_cases(
    reports: &mut Vec<JunitReport>,
    visible_sources: &HashMap<ProjectPath, u64>,
) {
    for report in reports.iter_mut() {
        let scope_path = &report.scope_path;
        let module_path = &report.module_path;
        report.cases.retain(|case| {
            source_path_for_case(scope_path, module_path, case)
                .is_some_and(|path| visible_sources.contains_key(&path))
        });
    }
    reports.retain(|report| !report.cases.is_empty());
}

fn source_path_for_case(
    scope_path: &ProjectPath,
    module_path: &str,
    case: &JunitCase,
) -> Option<ProjectPath> {
    let class_name = case.class_name.split('$').next()?;
    let module = if module_path == "." {
        "".to_string()
    } else {
        format!("{module_path}/")
    };
    let source = format!(
        "{module}src/test/java/{}.java",
        class_name.replace('.', "/")
    );
    let path = RelPath::from_unix_str(&source).ok()?;
    Some(ProjectPath {
        worktree_id: scope_path.worktree_id,
        path: path.into(),
    })
}

async fn verify_report_sources(
    reports: &[JunitReport],
    visible_sources: &HashMap<ProjectPath, u64>,
    buffer_store: &Entity<BufferStore>,
    cx: &mut gpui::AsyncApp,
) -> (HashSet<JvmVerifiedCase>, usize) {
    let mut cases_by_source: HashMap<ProjectPath, Vec<(usize, usize)>> = HashMap::new();
    for (report_index, report) in reports.iter().enumerate() {
        for (case_index, case) in report.cases.iter().enumerate() {
            if let Some(source_path) =
                source_path_for_case(&report.scope_path, &report.module_path, case)
            {
                cases_by_source
                    .entry(source_path)
                    .or_default()
                    .push((report_index, case_index));
            }
        }
    }
    let mut sources: Vec<_> = cases_by_source.into_iter().collect();
    sources.sort_by(|left, right| {
        left.0
            .worktree_id
            .cmp(&right.0.worktree_id)
            .then_with(|| left.0.path.cmp(&right.0.path))
    });
    let mut remaining_bytes = MAX_JUNIT_SOURCE_TOTAL_BYTES;
    let mut verified_cases = HashSet::new();
    let mut opened_sources = 0usize;
    let mut skipped_sources = 0usize;
    let parse_deadline = cx
        .background_executor()
        .timer(Duration::from_secs(5))
        .boxed()
        .shared();
    for (source_path, case_indices) in sources {
        if opened_sources >= MAX_JUNIT_SOURCE_FILES {
            skipped_sources += 1;
            continue;
        }
        let Some(size) = visible_sources.get(&source_path) else {
            skipped_sources += 1;
            continue;
        };
        let Ok(size) = usize::try_from(*size) else {
            skipped_sources += 1;
            continue;
        };
        if size > MAX_JUNIT_SOURCE_BYTES || size > remaining_bytes {
            skipped_sources += 1;
            continue;
        }
        let buffer = buffer_store.update(cx, |store, cx| {
            store.open_buffer_with_limit(source_path.clone(), MAX_JUNIT_SOURCE_BYTES, cx)
        });
        let buffer = match buffer.await {
            Ok(buffer) => buffer,
            Err(error) => {
                skipped_sources += 1;
                log::warn!("Could not verify a Java test source: {error:#}");
                continue;
            }
        };
        opened_sources += 1;
        remaining_bytes -= size;
        let version = buffer.read_with(cx, |buffer, _| buffer.version());
        let pending_parse = cx.update(|cx| {
            let source = buffer.read(cx);
            if (source.language().is_some() && *source.parse_status().borrow() == ParseStatus::Idle)
                || source
                    .language_registry()
                    .is_none_or(|languages| languages.available_language_for_name("Java").is_none())
            {
                return None;
            }
            let (sender, receiver) = oneshot::channel();
            let mut sender = Some(sender);
            let subscription = cx.observe(&buffer, move |buffer, cx| {
                let buffer = buffer.read(cx);
                if buffer.language().is_some()
                    && *buffer.parse_status().borrow() == ParseStatus::Idle
                    && let Some(sender) = sender.take()
                    && sender.send(()).is_err()
                {
                    log::debug!("Java source readiness observer was dropped");
                }
            });
            Some((receiver, subscription))
        });
        if let Some((ready, subscription)) = pending_parse {
            let ready = select(ready, parse_deadline.clone()).await;
            drop(subscription);
            if !matches!(ready, Either::Left((Ok(()), _))) {
                skipped_sources += 1;
                log::warn!(
                    "Java source syntax is not ready; wait for language loading and refresh Tests"
                );
                continue;
            }
        }
        if let Err(error) = buffer_store
            .update(cx, |store, cx| {
                store.open_buffer_with_limit(source_path.clone(), MAX_JUNIT_SOURCE_BYTES, cx)
            })
            .await
        {
            skipped_sources += 1;
            log::warn!("Could not revalidate a Java test source: {error:#}");
            continue;
        }
        if buffer.read_with(cx, |buffer, _| {
            buffer.version() != version
                || buffer.is_dirty()
                || *buffer.parse_status().borrow() != ParseStatus::Idle
        }) {
            skipped_sources += 1;
            continue;
        }
        let snapshot = buffer.read_with(cx, |buffer, _| buffer.snapshot());
        for runnable in snapshot.runnable_ranges(0..snapshot.len()) {
            for (report_index, case_index) in &case_indices {
                let report = &reports[*report_index];
                let case = &report.cases[*case_index];
                if let Some(verified) = verify_junit_runnable(report, case, &source_path, &runnable)
                {
                    verified_cases.insert(verified);
                }
            }
        }
    }
    (verified_cases, skipped_sources)
}

fn validate_module_binding(
    scope_path: &ProjectPath,
    module: &str,
    build_tool: JvmTestBuildTool,
) -> Result<()> {
    ensure!(
        module == "."
            || (!module.is_empty()
                && module.split('/').all(|part| {
                    !part.is_empty() && part != "." && part != ".." && !part.contains(['\\', ':'])
                })),
        "The selected Java test module is not a relative project path"
    );
    let expected_manifest = |name: &str| {
        if module == "." {
            name.to_string()
        } else {
            format!("{module}/{name}")
        }
    };
    let manifest = scope_path.path.as_unix_str();
    let matches = match build_tool {
        JvmTestBuildTool::Maven => manifest == expected_manifest("pom.xml"),
        JvmTestBuildTool::Gradle => {
            manifest == expected_manifest("build.gradle")
                || manifest == expected_manifest("build.gradle.kts")
        }
    };
    ensure!(
        matches,
        "The selected Java test is not bound to its module manifest"
    );
    Ok(())
}

pub async fn collect_junit_reports(
    fs: &dyn Fs,
    worktree_root: &Path,
    scope_path: ProjectPath,
    module_path: &str,
    build_tool: JvmTestBuildTool,
) -> Result<Vec<JunitReport>> {
    Ok(collect_junit_reports_with_budget(
        fs,
        worktree_root,
        scope_path,
        module_path,
        build_tool,
        MAX_JUNIT_DISCOVERY_BYTES,
    )
    .await?
    .0)
}

pub fn matching_junit_case<'a>(
    reports: &'a [JunitReport],
    selector: &JvmTestSelector,
) -> Result<(&'a JunitCase, &'a JunitReportVersion)> {
    let mut matches = reports
        .iter()
        .filter(|report| {
            report.scope_path == selector.scope_path && report.module_path == selector.module_path
        })
        .flat_map(|report| {
            report
                .cases
                .iter()
                .filter(move |case| {
                    case.class_name == selector.class_name
                        && simple_method_name(&case.display_name)
                            == Some(selector.method_name.as_str())
                })
                .map(move |case| (case, report.version.as_ref()))
        });
    let (case, version) = matches
        .next()
        .context("The selected Java test has no matching JUnit report")?;
    ensure!(
        matches.next().is_none(),
        "The selected Java test has ambiguous JUnit reports"
    );
    let version = version.context("The selected JUnit report has no host file version")?;
    Ok((case, version))
}

pub fn fresh_junit_case<'a>(
    reports: &'a [JunitReport],
    selector: &JvmTestSelector,
    previous: &JunitReportVersion,
) -> Result<&'a JunitCase> {
    let (case, current) = matching_junit_case(reports, selector)?;
    ensure!(
        current != previous,
        "The selected Java test has no new JUnit report from this Task"
    );
    Ok(case)
}

async fn collect_junit_reports_with_budget(
    fs: &dyn Fs,
    worktree_root: &Path,
    scope_path: ProjectPath,
    module_path: &str,
    build_tool: JvmTestBuildTool,
    byte_budget: usize,
) -> Result<(Vec<JunitReport>, usize)> {
    validate_module_binding(&scope_path, module_path, build_tool)?;
    let root_metadata = fs
        .metadata(worktree_root)
        .await?
        .context("The Java test worktree root is unavailable")?;
    ensure!(
        root_metadata.is_dir && !root_metadata.is_symlink,
        "The Java test worktree root is not a directory"
    );
    let report_subdirectory = match build_tool {
        JvmTestBuildTool::Maven => "target/surefire-reports",
        JvmTestBuildTool::Gradle => "build/test-results/test",
    };
    let relative_directory = if module_path == "." {
        PathBuf::from(report_subdirectory)
    } else {
        Path::new(module_path).join(report_subdirectory)
    };
    let report_directory = worktree_root.join(&relative_directory);
    let Some(directory_metadata) = fs.metadata(&report_directory).await? else {
        return Ok((Vec::new(), 0));
    };
    ensure!(
        directory_metadata.is_dir && !directory_metadata.is_symlink,
        "The Java test report directory is not a regular directory"
    );
    let mut files = Vec::new();
    let mut entries = fs.read_dir(&report_directory).await?;
    while let Some(entry) = entries.next().await {
        let file = entry?;
        ensure!(
            file.parent() == Some(report_directory.as_path()),
            "The Java test report directory returned an out-of-scope entry"
        );
        let Some(name) = file.file_name().and_then(|name| name.to_str()) else {
            continue;
        };
        if name.starts_with("TEST-") && name.ends_with(".xml") {
            files.push(file);
            ensure!(
                files.len() <= MAX_JUNIT_REPORT_FILES,
                "Java test discovery has too many JUnit report files"
            );
        }
    }
    files.sort();
    let mut total_bytes = 0usize;
    let mut total_read_bytes = 0usize;
    let mut reports = Vec::with_capacity(files.len());
    for file in files {
        let metadata = fs
            .metadata(&file)
            .await?
            .context("A Java test report disappeared during discovery")?;
        ensure!(
            !metadata.is_dir && !metadata.is_symlink && !metadata.is_fifo,
            "A Java test report is not a regular file"
        );
        let size = usize::try_from(metadata.len).context("A Java test report is too large")?;
        ensure!(
            size <= MAX_JUNIT_REPORT_BYTES,
            "A Java test report is too large"
        );
        total_bytes = total_bytes
            .checked_add(size)
            .context("Java test discovery report size overflow")?;
        ensure!(
            total_bytes <= byte_budget,
            "Java test discovery reports exceed the size limit"
        );
        let relative_file = relative_directory.join(
            file.file_name()
                .context("A Java test report has no file name")?,
        );
        let mut reader = fs
            .open_sync_confined_file(
                worktree_root,
                &relative_file,
                root_metadata.inode,
                metadata.inode,
            )
            .await?;
        let mut bytes = Vec::with_capacity(size);
        reader
            .by_ref()
            .take((MAX_JUNIT_REPORT_BYTES + 1) as u64)
            .read_to_end(&mut bytes)?;
        ensure!(
            bytes.len() <= MAX_JUNIT_REPORT_BYTES,
            "A Java test report changed beyond the size limit"
        );
        total_read_bytes = total_read_bytes
            .checked_add(bytes.len())
            .context("Java test discovery report size overflow")?;
        ensure!(
            total_read_bytes <= byte_budget,
            "Java test discovery reports exceed the size limit"
        );
        let mut report = parse_junit_report(&bytes, scope_path.clone(), module_path)?;
        report.version = Some(JunitReportVersion {
            relative_path: relative_file,
            inode: metadata.inode,
            mtime: metadata.mtime,
            size: metadata.len,
            digest: Sha256::digest(&bytes).into(),
        });
        reports.push(report);
    }
    Ok((reports, total_read_bytes))
}

fn ensure_jvm_test_protocol(version: u32) -> Result<()> {
    ensure!(
        version == JVM_TEST_DISCOVERY_PROTOCOL_VERSION,
        "Java test action protocol mismatch"
    );
    Ok(())
}

fn validate_remote_jvm_field(name: &str, value: &str) -> Result<()> {
    ensure!(
        !value.is_empty() && value.len() <= MAX_REMOTE_JVM_FIELD_BYTES && !value.contains('\0'),
        "{name} is invalid or exceeds the supported limit"
    );
    Ok(())
}

fn validate_remote_jvm_template(template: &TaskTemplate) -> Result<()> {
    validate_remote_jvm_field("Java test task label", &template.label)?;
    ensure!(
        matches!(
            template.command.as_str(),
            "./mvnw" | ".\\mvnw.cmd" | "./gradlew" | ".\\gradlew.bat"
        ) && template.args.len() <= 16,
        "The remote Java test action has an unsupported wrapper or arguments"
    );
    for argument in &template.args {
        validate_remote_jvm_field("Java test task argument", argument)?;
    }
    let expected = TaskTemplate {
        label: template.label.clone(),
        command: template.command.clone(),
        args: template.args.clone(),
        cwd: Some("$ZED_WORKTREE_ROOT".to_string()),
        show_summary: true,
        ..TaskTemplate::default()
    };
    ensure!(
        template == &expected,
        "The remote Java test action has unsupported Task fields"
    );
    Ok(())
}

fn remote_jvm_lifecycle_to_proto(
    project_id: u64,
    plan: &JvmRemoteTestActionPlan,
    event: &StructuredTaskLifecycleEvent,
    worktree_ids: Vec<u64>,
) -> proto::UpdateJvmTestRun {
    let (state, terminal_id, exit_code, success, termination_confirmed, diagnostic) =
        match &event.state {
            StructuredTaskState::Queued => (
                proto::JvmTestTaskState::Queued,
                None,
                None,
                false,
                false,
                None,
            ),
            StructuredTaskState::Running { terminal_id } => (
                proto::JvmTestTaskState::Running,
                terminal_id.map(|terminal_id| terminal_id.0),
                None,
                false,
                false,
                None,
            ),
            StructuredTaskState::Completed {
                terminal_id,
                exit_code,
                success,
            } => (
                proto::JvmTestTaskState::Completed,
                terminal_id.map(|terminal_id| terminal_id.0),
                *exit_code,
                *success,
                false,
                None,
            ),
            StructuredTaskState::SpawnError { .. } => (
                proto::JvmTestTaskState::SpawnError,
                None,
                None,
                false,
                false,
                Some(JVM_TASK_SPAWN_FAILURE.into()),
            ),
            StructuredTaskState::Cancelled {
                terminal_id,
                termination_confirmed,
            } => (
                proto::JvmTestTaskState::Cancelled,
                terminal_id.map(|terminal_id| terminal_id.0),
                None,
                false,
                *termination_confirmed,
                None,
            ),
        };
    proto::UpdateJvmTestRun {
        project_id,
        protocol_version: JVM_TEST_DISCOVERY_PROTOCOL_VERSION,
        discovery_generation: plan.discovery_generation.0,
        run_id: plan.run_id.0.clone(),
        node_id: plan.node_id.0.clone(),
        task_id: event.task_id.0.clone(),
        state: state as i32,
        terminal_id,
        exit_code,
        success,
        termination_confirmed,
        diagnostic,
        worktree_ids,
        abandon_unlaunched: false,
    }
}

fn remote_jvm_lifecycle_from_proto(
    request: &proto::UpdateJvmTestRun,
) -> Result<StructuredTaskLifecycleEvent> {
    validate_remote_jvm_field("Java test run ID", &request.run_id)?;
    validate_remote_jvm_field("Java test node ID", &request.node_id)?;
    validate_remote_jvm_field("Java test Task ID", &request.task_id)?;
    let terminal_id = request.terminal_id.map(StructuredTerminalId);
    let state = match proto::JvmTestTaskState::from_i32(request.state) {
        Some(proto::JvmTestTaskState::Queued) => StructuredTaskState::Queued,
        Some(proto::JvmTestTaskState::Running) => StructuredTaskState::Running { terminal_id },
        Some(proto::JvmTestTaskState::Completed) => StructuredTaskState::Completed {
            terminal_id,
            exit_code: request.exit_code,
            success: request.success,
        },
        Some(proto::JvmTestTaskState::SpawnError) => StructuredTaskState::SpawnError {
            // Older peers may still send raw Task errors containing private host details.
            message: JVM_TASK_SPAWN_FAILURE.into(),
        },
        Some(proto::JvmTestTaskState::Cancelled) => StructuredTaskState::Cancelled {
            terminal_id,
            termination_confirmed: request.termination_confirmed,
        },
        _ => bail!("Unsupported remote Java test lifecycle state"),
    };
    Ok(StructuredTaskLifecycleEvent {
        task_id: TaskId(request.task_id.clone()),
        state,
    })
}

pub fn selected_junit_task(
    selector: &JvmTestSelector,
    build_tool: JvmTestBuildTool,
    path_style: PathStyle,
) -> Result<TaskTemplate> {
    ensure!(
        selector
            .class_name
            .split('.')
            .all(|segment| { !segment.is_empty() && simple_method_name(segment) == Some(segment) }),
        "The selected Java test class has no stable runnable name"
    );
    ensure!(
        simple_method_name(&selector.method_name) == Some(selector.method_name.as_str()),
        "The selected Java test method has no stable runnable name"
    );
    let module = selector.module_path.as_str();
    validate_module_binding(&selector.scope_path, module, build_tool)?;
    let selected_test = format!("{}.{}", selector.class_name, selector.method_name);
    let (command, args) = match build_tool {
        JvmTestBuildTool::Maven => {
            let command = match path_style {
                PathStyle::Unix => "./mvnw",
                PathStyle::Windows => ".\\mvnw.cmd",
            };
            let mut args = Vec::new();
            if module != "." {
                args.extend([
                    "-pl".to_string(),
                    module.to_string(),
                    "-am".to_string(),
                    "-Dsurefire.failIfNoSpecifiedTests=false".to_string(),
                ]);
            }
            args.extend([
                format!("-Dtest={}#{}", selector.class_name, selector.method_name),
                "test".to_string(),
            ]);
            (command, args)
        }
        JvmTestBuildTool::Gradle => {
            let command = match path_style {
                PathStyle::Unix => "./gradlew",
                PathStyle::Windows => ".\\gradlew.bat",
            };
            let test_task = if module == "." {
                "test".to_string()
            } else {
                format!(":{}:test", module.replace('/', ":"))
            };
            (
                command,
                vec![
                    test_task,
                    "--tests".to_string(),
                    selected_test,
                    "--rerun-tasks".to_string(),
                ],
            )
        }
    };
    Ok(TaskTemplate {
        label: format!("Test {}#{}", selector.class_name, selector.method_name),
        command: command.to_string(),
        args,
        cwd: Some("$ZED_WORKTREE_ROOT".to_string()),
        show_summary: true,
        ..TaskTemplate::default()
    })
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct JvmVerifiedCase {
    pub scope_path: ProjectPath,
    pub module_path: String,
    pub class_name: String,
    pub display_name: String,
    pub method_name: String,
}

pub fn verify_junit_runnable(
    report: &JunitReport,
    case: &JunitCase,
    source_path: &ProjectPath,
    runnable: &RunnableRange,
) -> Option<JvmVerifiedCase> {
    if source_path.worktree_id != report.scope_path.worktree_id
        || runnable.runnable.language.name() != LanguageName::new("Java")
        || !runnable
            .runnable
            .tags
            .iter()
            .any(|tag| tag.0.as_ref() == "java-test-method")
        || runnable.extra_captures.get("annotation_name")?.as_str() != "Test"
    {
        return None;
    }
    let package = runnable
        .extra_captures
        .get("java_package_name")
        .map(String::as_str)
        .unwrap_or("");
    let class = runnable.extra_captures.get("java_class_name")?;
    let method = runnable.extra_captures.get("java_method_name")?;
    if !(package.is_empty()
        || package
            .split('.')
            .all(|segment| !segment.is_empty() && simple_method_name(segment) == Some(segment)))
        || simple_method_name(class) != Some(class.as_str())
        || simple_method_name(method) != Some(method.as_str())
        || simple_method_name(&case.display_name) != Some(method.as_str())
    {
        return None;
    }
    let qualified_class = if package.is_empty() {
        class.clone()
    } else {
        format!("{package}.{class}")
    };
    if case.class_name != qualified_class {
        return None;
    }
    let module_prefix = if report.module_path == "." {
        String::new()
    } else {
        format!("{}/", report.module_path)
    };
    let package_path = if package.is_empty() {
        String::new()
    } else {
        format!("{}/", package.replace('.', "/"))
    };
    let expected_source = format!("{module_prefix}src/test/java/{package_path}{class}.java");
    if source_path.path.as_unix_str() != expected_source {
        return None;
    }
    Some(JvmVerifiedCase {
        scope_path: report.scope_path.clone(),
        module_path: report.module_path.clone(),
        class_name: qualified_class,
        display_name: case.display_name.clone(),
        method_name: method.clone(),
    })
}

pub struct JvmTestDiscovery {
    pub snapshot: StructuredProviderSnapshot,
    pub selectors: HashMap<StructuredNodeId, JvmTestSelector>,
    pub results: Vec<(StructuredNodeId, StructuredNodeState, Option<String>)>,
}

pub fn parse_junit_report(
    bytes: &[u8],
    scope_path: ProjectPath,
    module_path: &str,
) -> Result<JunitReport> {
    if bytes.len() > MAX_JUNIT_REPORT_BYTES {
        bail!("JUnit report exceeds {MAX_JUNIT_REPORT_BYTES} bytes");
    }
    if module_path.len() > MAX_JUNIT_FIELD_BYTES {
        bail!("JUnit module path exceeds {MAX_JUNIT_FIELD_BYTES} bytes");
    }
    let mut reader = Reader::from_reader(bytes);
    let mut buffer = Vec::new();
    let mut depth = 0;
    let mut suite_name = None;
    let mut suite_depth = None;
    let mut case = None;
    let mut cases = Vec::new();
    let mut saw_suite = false;
    loop {
        match reader.read_event_into(&mut buffer)? {
            Event::Start(element) => {
                depth += 1;
                if depth > MAX_JUNIT_XML_DEPTH {
                    bail!("JUnit report XML nesting exceeds {MAX_JUNIT_XML_DEPTH}");
                }
                match element.name().as_ref() {
                    b"testsuite" => {
                        if suite_depth.is_some() {
                            bail!("nested JUnit testsuites are unsupported");
                        }
                        saw_suite = true;
                        suite_name = attribute(&element, b"name", &reader)?;
                        suite_depth = Some(depth);
                    }
                    b"testcase" => {
                        if suite_depth.is_none() || case.is_some() {
                            bail!("JUnit testcase is outside a suite or nested");
                        }
                        case = Some(parse_case(&element, suite_name.as_deref(), &reader)?);
                    }
                    b"failure" | b"error" if case.is_some() => {
                        let current = case.as_mut().context("JUnit case disappeared")?;
                        current.state = StructuredNodeState::Failed;
                        current.message = attribute(&element, b"message", &reader)?;
                    }
                    b"skipped" if case.is_some() => {
                        case.as_mut().context("JUnit case disappeared")?.state =
                            StructuredNodeState::Skipped;
                    }
                    _ => {}
                }
            }
            Event::Empty(element) => match element.name().as_ref() {
                b"testsuite" => {
                    if suite_depth.is_some() {
                        bail!("nested JUnit testsuites are unsupported");
                    }
                    saw_suite = true;
                }
                b"testcase" => {
                    if suite_depth.is_none() || case.is_some() {
                        bail!("JUnit testcase is outside a suite or nested");
                    }
                    cases.push(parse_case(&element, suite_name.as_deref(), &reader)?);
                    if cases.len() > MAX_JUNIT_CASES {
                        bail!("JUnit report contains more than {MAX_JUNIT_CASES} cases");
                    }
                }
                b"failure" | b"error" if case.is_some() => {
                    let current = case.as_mut().context("JUnit case disappeared")?;
                    current.state = StructuredNodeState::Failed;
                    current.message = attribute(&element, b"message", &reader)?;
                }
                b"skipped" if case.is_some() => {
                    case.as_mut().context("JUnit case disappeared")?.state =
                        StructuredNodeState::Skipped;
                }
                _ => {}
            },
            Event::End(element) => {
                match element.name().as_ref() {
                    b"testcase" => {
                        cases.push(case.take().context("JUnit testcase end without start")?);
                        if cases.len() > MAX_JUNIT_CASES {
                            bail!("JUnit report contains more than {MAX_JUNIT_CASES} cases");
                        }
                    }
                    b"testsuite" => {
                        if suite_depth != Some(depth) || case.is_some() {
                            bail!("JUnit testsuite ended before its cases");
                        }
                        suite_name = None;
                        suite_depth = None;
                    }
                    _ => {}
                }
                depth = depth
                    .checked_sub(1)
                    .context("JUnit XML end without start")?;
            }
            Event::DocType(_) => bail!("JUnit report DTD is unsupported"),
            Event::Eof => break,
            _ => {}
        }
        buffer.clear();
    }
    if !saw_suite || case.is_some() || suite_depth.is_some() || depth != 0 {
        bail!("JUnit report is incomplete or has no testsuite");
    }
    Ok(JunitReport {
        scope_path,
        module_path: module_path.to_string(),
        cases,
        version: None,
    })
}

fn parse_case(
    element: &BytesStart<'_>,
    suite_name: Option<&str>,
    reader: &Reader<&[u8]>,
) -> Result<JunitCase> {
    let class_name = attribute(element, b"classname", reader)?
        .or_else(|| suite_name.map(str::to_string))
        .context("JUnit testcase has no classname or suite name")?;
    let display_name =
        attribute(element, b"name", reader)?.context("JUnit testcase has no name")?;
    if class_name.is_empty() || display_name.is_empty() {
        bail!("JUnit testcase has an empty class or name");
    }
    Ok(JunitCase {
        class_name,
        display_name,
        state: StructuredNodeState::Passed,
        message: None,
    })
}

fn attribute(
    element: &BytesStart<'_>,
    key: &[u8],
    reader: &Reader<&[u8]>,
) -> Result<Option<String>> {
    for attribute in element.attributes() {
        let attribute = attribute?;
        if attribute.key.as_ref() == key {
            let value = attribute.decode_and_unescape_value(reader.decoder())?;
            if value.len() > MAX_JUNIT_FIELD_BYTES {
                bail!("JUnit XML field exceeds {MAX_JUNIT_FIELD_BYTES} bytes");
            }
            return Ok(Some(value.into_owned()));
        }
    }
    Ok(None)
}

impl JvmTestDiscovery {
    pub fn from_reports(
        reports: &[JunitReport],
        generation: DiscoveryGeneration,
        verified_cases: &HashSet<JvmVerifiedCase>,
    ) -> Result<Self> {
        let provider_id = StructuredProviderId(JVM_JUNIT_PROVIDER_ID.to_string());
        let root_id = StructuredNodeId(JVM_JUNIT_PROVIDER_ID.to_string());
        let mut nodes = vec![StructuredNode {
            experimental_feature: None,
            experimental_feature_enabled: None,
            debug_capability: None,
            id: root_id.clone(),
            parent_id: None,
            label: "Java JUnit tests".to_string(),
            kind: StructuredNodeKind::Provider,
            path: None,
        }];
        let mut selectors = HashMap::new();
        let mut results = Vec::new();
        let mut groups = BTreeMap::new();
        let mut suites = BTreeMap::new();
        let mut counts = HashMap::new();
        for report in reports {
            for case in &report.cases {
                *counts
                    .entry((
                        &report.scope_path,
                        &report.module_path,
                        &case.class_name,
                        &case.display_name,
                    ))
                    .or_insert(0usize) += 1;
            }
        }
        let mut occurrences = HashMap::new();
        for report in reports {
            let group_key = (&report.scope_path, report.module_path.as_str());
            let group_id = groups
                .entry(group_key)
                .or_insert_with(|| {
                    let id = node_id(&[
                        "module",
                        &report.scope_path.worktree_id.to_string(),
                        report.scope_path.path.as_unix_str(),
                        &report.module_path,
                    ]);
                    nodes.push(StructuredNode {
                        experimental_feature: None,
                        experimental_feature_enabled: None,
                        debug_capability: None,
                        id: id.clone(),
                        parent_id: Some(root_id.clone()),
                        label: report.module_path.clone(),
                        kind: StructuredNodeKind::Group,
                        path: Some(report.scope_path.clone()),
                    });
                    id
                })
                .clone();
            for case in &report.cases {
                if results.len() >= MAX_JUNIT_CASES {
                    bail!("JUnit discovery contains more than {MAX_JUNIT_CASES} cases");
                }
                let suite_key = (
                    &report.scope_path,
                    report.module_path.as_str(),
                    case.class_name.as_str(),
                );
                let suite_id = suites
                    .entry(suite_key)
                    .or_insert_with(|| {
                        let id = node_id(&[
                            "suite",
                            &report.scope_path.worktree_id.to_string(),
                            report.scope_path.path.as_unix_str(),
                            &report.module_path,
                            &case.class_name,
                        ]);
                        nodes.push(StructuredNode {
                            experimental_feature: None,
                            experimental_feature_enabled: None,
                            debug_capability: None,
                            id: id.clone(),
                            parent_id: Some(group_id.clone()),
                            label: case.class_name.clone(),
                            kind: StructuredNodeKind::Suite,
                            path: None,
                        });
                        id
                    })
                    .clone();
                let key = (
                    &report.scope_path,
                    &report.module_path,
                    &case.class_name,
                    &case.display_name,
                );
                let occurrence = occurrences.entry(key).or_insert(0usize);
                let case_id = node_id(&[
                    "case",
                    &report.scope_path.worktree_id.to_string(),
                    report.scope_path.path.as_unix_str(),
                    &report.module_path,
                    &case.class_name,
                    &case.display_name,
                    &occurrence.to_string(),
                ]);
                *occurrence += 1;
                nodes.push(StructuredNode {
                    experimental_feature: None,
                    experimental_feature_enabled: None,
                    debug_capability: None,
                    id: case_id.clone(),
                    parent_id: Some(suite_id),
                    label: case.display_name.clone(),
                    kind: StructuredNodeKind::Case,
                    path: None,
                });
                if counts.get(&key) == Some(&1)
                    && let Some(method_name) = simple_method_name(&case.display_name)
                {
                    let verified_case = JvmVerifiedCase {
                        scope_path: report.scope_path.clone(),
                        module_path: report.module_path.clone(),
                        class_name: case.class_name.clone(),
                        display_name: case.display_name.clone(),
                        method_name: method_name.to_string(),
                    };
                    if verified_cases.contains(&verified_case) {
                        selectors.insert(
                            case_id.clone(),
                            JvmTestSelector {
                                scope_path: verified_case.scope_path,
                                module_path: verified_case.module_path,
                                class_name: verified_case.class_name,
                                method_name: verified_case.method_name,
                            },
                        );
                    }
                }
                results.push((case_id, case.state, case.message.clone()));
            }
        }
        let status = if results.is_empty() {
            StructuredProviderStatus::Empty
        } else {
            StructuredProviderStatus::Current
        };
        let snapshot =
            StructuredProviderSnapshot::discovery(provider_id, generation, status, nodes);
        Ok(Self {
            snapshot,
            selectors,
            results,
        })
    }
}

fn simple_method_name(display_name: &str) -> Option<&str> {
    let method = display_name.strip_suffix("()").unwrap_or(display_name);
    let mut chars = method.chars();
    let first = chars.next()?;
    if !(first.is_ascii_alphabetic() || first == '_' || first == '$') {
        return None;
    }
    chars
        .all(|character| character.is_ascii_alphanumeric() || character == '_' || character == '$')
        .then_some(method)
}

fn node_id(parts: &[&str]) -> StructuredNodeId {
    let mut hasher = Sha256::new();
    for part in parts {
        hasher.update((part.len() as u64).to_be_bytes());
        hasher.update(part.as_bytes());
    }
    StructuredNodeId(format!("jvm-junit:{:x}", hasher.finalize()))
}

#[cfg(test)]
#[test]
fn remote_java_spawn_errors_do_not_export_raw_task_diagnostics() {
    let raw_message = "PRIVATE_SPAWN_DETAIL /outside/private/runtime token=secret 🚫".repeat(20);
    let event = StructuredTaskLifecycleEvent {
        task_id: TaskId("java-spawn-error".into()),
        state: StructuredTaskState::SpawnError {
            message: raw_message.clone(),
        },
    };
    let plan = JvmRemoteTestActionPlan {
        run_id: StructuredRunId("java-run".into()),
        worktree_id: WorktreeId::from_usize(1),
        discovery_generation: DiscoveryGeneration(2),
        node_id: StructuredNodeId("java-case".into()),
        template: TaskTemplate::default(),
        client_generation: 1,
        report_maximum_bytes: None,
        cancellation: None,
    };
    let mut request = remote_jvm_lifecycle_to_proto(42, &plan, &event, vec![1]);
    let safe_message =
        "The Java test Task failed to start; inspect the Task terminal and project-host logs";
    assert_eq!(request.diagnostic.as_deref(), Some(safe_message));
    assert!(!format!("{request:?}").contains("PRIVATE_SPAWN_DETAIL"));
    assert!(
        matches!(&event.state, StructuredTaskState::SpawnError { message } if message == &raw_message)
    );
    for diagnostic in [Some(raw_message), None, Some(String::new())] {
        request.diagnostic = diagnostic;
        let decoded =
            remote_jvm_lifecycle_from_proto(&request).expect("valid legacy Java lifecycle");
        assert!(
            matches!(decoded.state, StructuredTaskState::SpawnError { message } if message == safe_message)
        );
    }
}

fn validate_remote_report_limit(requested: Option<usize>, acknowledged: Option<u64>) -> Result<()> {
    ensure!(
        acknowledged == requested.map(|maximum| maximum as u64),
        "The project host did not acknowledge the exact Java report byte limit; reconnect to a compatible host"
    );
    Ok(())
}

#[cfg(test)]
#[test]
fn doctor_jvm_reproduction_missing_or_changed_acknowledgement_is_blocked() {
    assert!(validate_remote_report_limit(None, None).is_ok());
    assert!(validate_remote_report_limit(Some(5 * 1024 * 1024), Some(5 * 1024 * 1024)).is_ok());
    assert!(validate_remote_report_limit(Some(5 * 1024 * 1024), None).is_err());
    assert!(validate_remote_report_limit(Some(5 * 1024 * 1024), Some(16 * 1024 * 1024)).is_err());
    assert!(validate_remote_report_limit(None, Some(5 * 1024 * 1024)).is_err());
}

fn validate_jvm_action_protocol(version: u32, bounded: bool) -> Result<()> {
    ensure!(
        version
            == if bounded {
                JVM_TEST_BOUNDED_ACTION_PROTOCOL_VERSION
            } else {
                JVM_TEST_DISCOVERY_PROTOCOL_VERSION
            },
        "Java test action protocol mismatch; bounded inspection requires a compatible host before report reads"
    );
    Ok(())
}

#[cfg(test)]
#[test]
fn doctor_jvm_reproduction_old_action_protocol_cannot_collect_bounded_reports() {
    assert!(validate_jvm_action_protocol(1, false).is_ok());
    assert!(validate_jvm_action_protocol(2, true).is_ok());
    assert!(validate_jvm_action_protocol(1, true).is_err());
    assert!(validate_jvm_action_protocol(2, false).is_err());
}
